#!/usr/bin/env python3
"""
das_torch详细注释版本
原文件：das_torch.py
作者：Dongwoon Hyun (dongwoon.hyun@stanford.edu)
创建日期：2020-03-09
作用：PyTorch实现的DAS（延迟叠加）波束成形算法，支持平面波和聚焦发射
"""

import torch
from torch.nn.functional import grid_sample

PI = 3.14159265359  # 圆周率常数


class DAS_PW(torch.nn.Module):
    """ 
    PyTorch实现的DAS平面波波束成形
    
    DAS (Delay-And-Sum) 是最经典的超声波束成形算法：
    1. 计算每个像素点到各个阵元的声波传播延迟
    2. 根据延迟对接收信号进行时间校正
    3. 将校正后的信号叠加得到重建图像
    
    这个类将DAS实现为PyTorch神经网络模块，使得：
    - 可以在GPU上加速计算
    - 参数可以设为可训练（用于学习波束成形）
    - 可以与深度学习模型集成
    """

    def __init__(
        self,
        P,                                    # PlaneWaveData对象，包含采集参数
        grid,                                 # 重建网格，形状为[ncols, nrows, 3]
        ang_list=None,                        # 使用的角度列表
        ele_list=None,                        # 使用的阵元列表  
        rxfnum=2,                            # 接收F数（控制接收孔径）
        dtype=torch.float,                    # 数据类型
        device=torch.device("cuda:0"),        # 计算设备（GPU/CPU）
    ):
        """ 
        DAS_PW初始化方法
        
        参数说明：
        P: PlaneWaveData对象，包含：
           - angles: 发射角度 [弧度]
           - ele_pos: 阵元位置 [米]
           - fc, fs, fdemod: 频率参数 [Hz]
           - c: 声速 [m/s]
           - time_zero: 时间零点 [秒]
           
        grid: 重建网格，每个点包含(x,y,z)坐标 [米]
        ang_list: 选择使用的角度索引（None表示使用全部角度）
        ele_list: 选择使用的阵元索引（None表示使用全部阵元）
        rxfnum: 接收F数，控制动态接收聚焦的孔径大小
        """
        super().__init__()
        
        # ===== 处理角度和阵元列表 =====
        # 如果没有指定，则使用全部角度和阵元
        if ang_list is None:
            ang_list = range(P.angles.shape[0])  # 使用所有角度
        elif not hasattr(ang_list, "__getitem__"):  # 如果是单个值，转为列表
            ang_list = [ang_list]
            
        if ele_list is None:
            ele_list = range(P.ele_pos.shape[0])  # 使用所有阵元
        elif not hasattr(ele_list, "__getitem__"):  # 如果是单个值，转为列表
            ele_list = [ele_list]

        # ===== 将PlaneWaveData转换为PyTorch张量 =====
        # 这些是超声成像的物理参数，存储为GPU张量以加速计算
        self.angles = torch.tensor(P.angles, dtype=dtype, device=device)      # 发射角度
        self.ele_pos = torch.tensor(P.ele_pos, dtype=dtype, device=device)    # 阵元位置
        self.fc = torch.tensor(P.fc, dtype=dtype, device=device)              # 中心频率
        self.fs = torch.tensor(P.fs, dtype=dtype, device=device)              # 采样频率
        self.fdemod = torch.tensor(P.fdemod, dtype=dtype, device=device)      # 解调频率
        self.c = torch.tensor(P.c, dtype=dtype, device=device)                # 声速
        self.time_zero = torch.tensor(P.time_zero, dtype=dtype, device=device)  # 时间零点

        # ===== 转换重建网格 =====
        self.grid = torch.tensor(grid, dtype=dtype, device=device).view(-1, 3)  # 展平为[npixels, 3]
        self.out_shape = grid.shape[:-1]  # 输出图像形状，例如(508, 387)

        # ===== 存储其他配置信息 =====
        self.ang_list = torch.tensor(ang_list, dtype=torch.long, device=device)  # 使用的角度索引
        self.ele_list = torch.tensor(ele_list, dtype=torch.long, device=device)  # 使用的阵元索引
        self.dtype = dtype
        self.device = device

    def forward(self, x):
        """ 
        DAS_PW的前向传播函数 - 执行波束成形计算
        
        输入：
        x: 包含(idata, qdata)的元组
           idata: I分量数据，形状为[nangles, nchans, nsamps]
           qdata: Q分量数据，形状为[nangles, nchans, nsamps]
           
        输出：
        (idas, qdas): 重建后的I和Q分量图像
        """
        dtype, device = self.dtype, self.device

        # ===== 加载输入数据到GPU =====
        idata, qdata = x
        idata = torch.tensor(idata, dtype=dtype, device=device)
        qdata = torch.tensor(qdata, dtype=dtype, device=device)

        # ===== 计算几何延迟 =====
        nangles = len(self.ang_list)    # 使用的角度数
        nelems = len(self.ele_list)     # 使用的阵元数  
        npixels = self.grid.shape[0]    # 像素总数
        xlims = (self.ele_pos[0, 0], self.ele_pos[-1, 0])  # 阵列孔径宽度

        # 初始化延迟和权重数组
        txdel = torch.zeros((nangles, npixels), dtype=dtype, device=device)    # 发射延迟
        rxdel = torch.zeros((nelems, npixels), dtype=dtype, device=device)     # 接收延迟
        txapo = torch.ones((nangles, npixels), dtype=dtype, device=device)     # 发射权重
        rxapo = torch.ones((nelems, npixels), dtype=dtype, device=device)      # 接收权重

        # ===== 计算发射延迟和权重 =====
        for i, tx in enumerate(self.ang_list):
            # 计算平面波发射延迟：距离 = x*sin(θ) + z*cos(θ) 
            txdel[i] = delay_plane(self.grid, self.angles[[tx]])
            # 加上时间零点对应的距离偏移
            txdel[i] += self.time_zero[tx] * self.c
            # 计算发射权重（限制有效成像区域）
            txapo[i] = apod_plane(self.grid, self.angles[tx], xlims)

        # ===== 计算接收延迟和权重 =====
        for j, rx in enumerate(self.ele_list):
            # 计算聚焦接收延迟：每个像素到阵元的欧氏距离
            rxdel[j] = delay_focus(self.grid, self.ele_pos[[rx]])
            # 计算接收权重（动态聚焦，F数控制）
            rxapo[j] = apod_focus(self.grid, self.ele_pos[rx])

        # ===== 将距离延迟转换为采样点延迟 =====
        # 距离延迟[米] × 采样率[Hz] / 声速[m/s] = 采样点延迟
        txdel *= self.fs / self.c
        rxdel *= self.fs / self.c

        # ===== 初始化输出图像 =====
        idas = torch.zeros(npixels, dtype=self.dtype, device=self.device)  # I分量输出
        qdas = torch.zeros(npixels, dtype=self.dtype, device=self.device)  # Q分量输出

        # ===== 核心DAS循环：对所有发射角度和接收阵元进行延迟叠加 =====
        for t, td, ta in zip(self.ang_list, txdel, txapo):    # 遍历发射角度
            for r, rd, ra in zip(self.ele_list, rxdel, rxapo):  # 遍历接收阵元
                
                # === 获取当前发射-接收对的数据 ===
                # 从第t个发射角度、第r个接收阵元获取IQ数据
                iq = torch.stack((idata[t, r], qdata[t, r]), dim=0).view(1, 2, 1, -1)
                
                # === 计算总延迟并准备插值 ===
                delays = td + rd  # 总延迟 = 发射延迟 + 接收延迟
                
                # 将延迟转换为grid_sample所需的归一化坐标 [-1, 1]
                # 公式：(delay * 2 + 1) / nsamps - 1 将延迟映射到[-1,1]
                dgs = (delays.view(1, 1, -1, 1) * 2 + 1) / idata.shape[-1] - 1
                dgs = torch.cat((dgs, 0 * dgs), axis=-1)  # 添加第二维度坐标（为0）
                
                # === 使用双线性插值获取延迟校正后的数据 ===
                # grid_sample执行双线性插值，相当于对信号进行延迟校正
                ifoc, qfoc = grid_sample(iq, dgs).view(2, -1)
                
                # === 相位校正（仅对解调数据） ===
                if self.fdemod != 0:
                    # 计算由于几何延迟引起的相位偏移
                    tshift = delays.view(-1) / self.fs - self.grid[:, 2] * 2 / self.c
                    theta = 2 * PI * self.fdemod * tshift  # 相位角
                    ifoc, qfoc = _complex_rotate(ifoc, qfoc, theta)  # 相位旋转校正
                
                # === 应用权重并累加到输出 ===
                apods = ta * ra  # 组合发射和接收权重
                idas += ifoc * apods  # 加权累加I分量
                qdas += qfoc * apods  # 加权累加Q分量

        # ===== 恢复图像形状并返回 =====
        idas = idas.view(self.out_shape)  # 恢复为(height, width)
        qdas = qdas.view(self.out_shape)
        return idas, qdas


class DAS_FT(torch.nn.Module):
    """ 
    PyTorch实现的DAS聚焦发射波束成形
    
    与平面波DAS的区别：
    - 平面波：所有阵元同时发射，每次发射覆盖整个成像区域
    - 聚焦发射：能量聚焦到特定点，逐点扫描成像
    
    聚焦发射特点：
    - 每次发射只成像一条线或一个点
    - 需要多次发射完成整幅图像
    - 信噪比更高，但成像速度较慢
    """

    def __init__(
        self, 
        F,                                    # FocusedTxData对象
        grid,                                 # 重建网格
        rxfnum=2,                            # 接收F数
        dtype=torch.float,                    # 数据类型
        device=torch.device("cuda:0"),        # 计算设备
    ):
        """ 
        DAS_FT初始化方法
        
        参数：
        F: FocusedTxData对象，包含：
           - tx_ori: 发射起始点 [米]
           - tx_dir: 发射方向 [弧度]
           - ele_pos: 阵元位置 [米]  
           - fc, fs, fdemod, c, time_zero: 其他参数
        """
        super().__init__()

        # ===== 转换聚焦发射数据为PyTorch张量 =====
        self.tx_ori = torch.tensor(F.tx_ori, dtype=dtype, device=device)      # 发射起始点
        self.tx_dir = torch.tensor(F.tx_dir, dtype=dtype, device=device)      # 发射方向
        self.ele_pos = torch.tensor(F.ele_pos, dtype=dtype, device=device)    # 阵元位置
        self.fc = torch.tensor(F.fc, dtype=dtype, device=device)              # 中心频率
        self.fs = torch.tensor(F.fs, dtype=dtype, device=device)              # 采样频率
        self.fdemod = torch.tensor(F.fdemod, dtype=dtype, device=device)      # 解调频率
        self.c = torch.tensor(F.c, dtype=dtype, device=device)                # 声速
        self.time_zero = torch.tensor(F.time_zero, dtype=dtype, device=device)  # 时间零点

        # ===== 转换重建网格 =====
        self.grid = torch.tensor(grid, dtype=dtype, device=device)
        self.out_shape = grid.shape[:-1]

        # ===== 存储其他信息 =====
        self.dtype = dtype
        self.device = device
        self.rxfnum = torch.tensor(rxfnum)

    def forward(self, x):
        """ 
        DAS_FT前向传播 - 聚焦发射波束成形
        
        输入：
        x: (idata, qdata)，形状为[nxmits, nelems, nsamps]
           nxmits: 发射次数
           nelems: 阵元数
           nsamps: 采样点数
        """
        idata, qdata = x
        dtype, device = self.dtype, self.device
        nxmits, nelems, nsamps = idata.shape
        nx, nz = self.grid.shape[:2]

        # 初始化输出
        idas = torch.zeros((nx, nz), dtype=dtype, device=device)
        qdas = torch.zeros((nx, nz), dtype=dtype, device=device)

        # ===== 遍历所有发射 =====
        for t in range(nxmits):
            # 计算发射延迟：发射点到像素的距离
            txdel = torch.norm(self.grid[t] - self.tx_ori[t].unsqueeze(0), dim=-1)
            
            # 计算接收延迟：像素到各阵元的距离
            rxdel = delay_focus(self.grid[t].view(-1, 1, 3), self.ele_pos).T
            
            # 总延迟转换为采样点
            delays = ((txdel + rxdel) / self.c + self.time_zero[t]) * self.fs
            
            # 准备数据用于grid_sample
            iq = torch.stack((idata[t], qdata[t]), axis=0).unsqueeze(0)
            
            # 转换延迟为归一化坐标
            dgsz = (delays.unsqueeze(0) * 2 + 1) / idata.shape[-1] - 1
            dgsx = torch.arange(nelems, dtype=dtype, device=device)
            dgsx = ((dgsx * 2 + 1) / nelems - 1).view(1, -1, 1)
            dgsx = dgsx + 0 * dgsz  # 通过广播匹配形状
            dgs = torch.stack((dgsz, dgsx), axis=-1)
            
            # 执行插值
            ifoc, qfoc = grid_sample(iq, dgs, align_corners=False)[0]

            # 相位校正
            if self.fdemod != 0:
                tshift = delays / self.fs - self.grid[[t], :, 2] * 2 / self.c
                theta = 2 * PI * self.fdemod * tshift
                ifoc, qfoc = _complex_rotate(ifoc, qfoc, theta)

            # 计算权重
            apods = apod_focus(self.grid[t], self.ele_pos, fnum=self.rxfnum)

            # 应用权重并累加
            ifoc *= apods
            qfoc *= apods
            idas[t] = ifoc.sum(axis=0, keepdim=False)
            qdas[t] = qfoc.sum(axis=0, keepdim=False)

        return idas, qdas


def _complex_rotate(I, Q, theta):
    """
    复数相位旋转函数
    
    对复数信号 z = I + jQ 应用相位旋转：
    z' = z * exp(j*theta) = z * (cos(theta) + j*sin(theta))
    
    参数：
    I, Q: 复数的实部和虚部
    theta: 相位旋转角度 [弧度]
    
    返回：
    Ir, Qr: 旋转后的实部和虚部
    """
    # 复数乘法：(I + jQ) * (cos + j*sin)
    Ir = I * torch.cos(theta) - Q * torch.sin(theta)  # 实部
    Qr = Q * torch.cos(theta) + I * torch.sin(theta)  # 虚部
    return Ir, Qr


def delay_focus(grid, ele_pos):
    """
    计算聚焦延迟：像素点到阵元的欧氏距离
    
    参数：
    grid: 像素位置，形状为[npixels, 3]，包含(x,y,z)坐标
    ele_pos: 阵元位置，形状为[nelems, 3]，包含(x,y,z)坐标
    
    返回：
    dist: 距离矩阵，形状为[nelems, npixels]
    
    物理含义：
    声波从阵元传播到像素点的几何距离
    在接收时，表示声波从像素点反射回阵元的距离
    """
    # 使用广播计算距离：||grid - ele_pos||
    # grid[npixels, 3] - ele_pos[nelems, 1, 3] -> [nelems, npixels, 3]
    dist = torch.norm(grid - ele_pos.unsqueeze(0), dim=-1)
    return dist


def delay_plane(grid, angles):
    """
    计算平面波延迟：像素点沿波传播方向的投影距离
    
    参数：
    grid: 像素位置，形状为[npixels, 3]
    angles: 平面波角度，形状为[nangles]，单位为弧度
    
    返回：
    dist: 延迟距离，形状为[nangles, npixels] 
    
    物理含义：
    平面波传播方向为：direction = [sin(θ), 0, cos(θ)]
    延迟距离 = 像素位置在传播方向上的投影
             = x*sin(θ) + z*cos(θ)
    
    这个距离表示相对于参考点，声波到达该像素的额外传播距离
    """
    # 使用广播简化计算，增加角度维度
    x = grid[:, 0].unsqueeze(0)  # [1, npixels]
    z = grid[:, 2].unsqueeze(0)  # [1, npixels]  
    
    # 计算投影距离：x*sin(θ) + z*cos(θ)
    # angles[nangles] -> angles[nangles, 1] 通过广播得到 [nangles, npixels]
    # 确保angles有正确的维度进行unsqueeze
    if angles.dim() == 0:  # 标量
        angles = angles.unsqueeze(0)
    if angles.size(0) == 1 and angles.dim() == 1:  # 单个角度的1维张量
        angles_expanded = angles.unsqueeze(1)  # [1, 1]
    else:
        angles_expanded = angles.unsqueeze(-1)  # [nangles, 1]
    
    dist = x * torch.sin(angles_expanded) + z * torch.cos(angles_expanded)
    # print("1111111")
    return dist


def apod_focus(grid, ele_pos, fnum=1, min_width=1e-3):
    """
    计算聚焦接收权重（动态聚焦）
    
    参数：
    grid: 像素位置 [npixels, 3]
    ele_pos: 阵元位置 [nelems, 3] 
    fnum: F数，控制接收孔径大小
    min_width: 最小孔径宽度 [米]
    
    返回：
    apod: 权重矩阵 [nelems, npixels]
    
    物理含义：
    F数 = 聚焦深度 / 孔径直径
    较大的F数 -> 较小的孔径 -> 更好的横向分辨率，但信噪比较低
    较小的F数 -> 较大的孔径 -> 更好的信噪比，但横向分辨率较差
    
    动态聚焦：对不同深度的像素使用不同的接收孔径
    浅层像素使用较小孔径，深层像素使用较大孔径
    """
    # 通过广播计算阵元到像素的向量
    ppos = grid.unsqueeze(0)        # [1, npixels, 3]
    epos = ele_pos.view(-1, 1, 3)   # [nelems, 1, 3]
    v = ppos - epos                 # [nelems, npixels, 3]
    
    # 计算有效F数：深度/横向距离 = z/x
    # 选择F数大于设定值的阵元-像素对（即横向距离足够小）
    mask = torch.abs(v[:, :, 2] / v[:, :, 0]) > fnum
    
    # 保留最小孔径内的阵元（避免孔径过小）
    mask = mask | (torch.abs(v[:, :, 0]) <= min_width)
    
    # 处理孔径边缘情况
    mask = mask | ((v[:, :, 0] >= min_width) & (ppos[:, :, 0] <= epos[0, 0, 0]))
    mask = mask | ((v[:, :, 0] <= -min_width) & (ppos[:, :, 0] >= epos[-1, 0, 0]))
    
    # 转换为浮点权重
    apod = mask.float()
    return apod


def apod_plane(grid, angles, xlims):
    """
    计算平面波发射权重
    
    参数：
    grid: 像素位置 [npixels, 3]
    angles: 平面波角度 [nangles]
    xlims: 孔径横向边界 [xmin, xmax]
    
    返回：  
    apod: 权重矩阵 [nangles, npixels]
    
    物理含义：
    限制有效成像区域，只保留能被阵列有效照射的像素
    
    方法：将像素沿发射角度投影回阵列平面
    - 如果投影点在阵列范围内，则权重为1
    - 否则权重为0
    
    这避免了阵列外侧区域的成像伪影
    """
    pix = grid.unsqueeze(0)         # [1, npixels, 3]
    ang = angles.view(-1, 1, 1)     # [nangles, 1, 1]
    
    # 将像素沿发射角度投影回阵列平面
    # 投影公式：x_proj = x - z * tan(θ)
    x_proj = pix[:, :, 0] - pix[:, :, 2] * torch.tan(ang)
    
    # 选择投影点在阵列范围内的像素（带1.2倍安全系数）
    mask = (x_proj >= xlims[0] * 1.2) & (x_proj <= xlims[1] * 1.2)
    
    # 转换为浮点权重
    apod = mask.float()
    return apod


# ===== 总结和使用说明 =====
"""
DAS波束成形算法总结：

1. 核心思想：
   - 延迟校正：补偿声波传播的几何延迟
   - 相干叠加：将校正后的信号相加增强信号
   - 权重控制：优化成像质量和减少伪影

2. 关键步骤：
   a) 计算几何延迟（距离/声速）
   b) 转换为采样延迟（×采样率）
   c) 插值获取延迟校正后的信号
   d) 相位校正（仅解调数据）
   e) 应用权重并累加

3. 两种发射模式：
   - 平面波(DAS_PW)：快速成像，角度复合
   - 聚焦发射(DAS_FT)：高质量，逐点扫描

4. 在AUGAN中的作用：
   - DAS重建单角度图像（低质量输入）
   - DAS重建多角度图像（高质量目标）
   - GAN学习从低质量到高质量的映射

使用示例：
```python
# 创建DAS网络
das = DAS_PW(plane_wave_data, grid, ang_list=[37])  # 单角度

# 执行波束成形
idas, qdas = das((idata, qdata))

# 计算图像强度
img = torch.sqrt(idas**2 + qdas**2)
```
"""