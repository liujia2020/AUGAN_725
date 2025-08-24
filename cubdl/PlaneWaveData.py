#!/usr/bin/env python3
"""
PlaneWaveData详细注释版本
原文件：PlaneWaveData.py
作用：定义平面波超声数据的标准格式和PICMUS数据加载器
"""

import numpy as np
import h5py
from scipy.signal import hilbert


class PlaneWaveData:
    """ 
    平面波超声数据的模板类
    
    PlaneWaveData是一个容器/数据类，保存描述平面波采集的所有信息。
    用户应该创建子类并根据自己的数据存储方式重新实现__init__()方法。
    
    必需的信息包括：
    idata       同相(实部)数据，形状为 (n角度, n通道, n采样点)
    qdata       正交(虚部)数据，形状为 (n角度, n通道, n采样点)
    angles      角度列表 [弧度]
    ele_pos     传感器位置，形状为 (N,3) [米]
    fc          中心频率 [Hz]
    fs          采样频率 [Hz]
    fdemod      解调频率(如果数据已解调) [Hz]
    c           声速 [m/s]
    time_zero   每次采集的时间零点列表 [秒]
    
    正确的实现可以通过validate()方法检查。
    参见PICMUSData类获取完整实现的例子。
    """

    def __init__(self):
        """ 
        用户必须重新实现此函数来加载自己的数据
        
        注意：不要直接使用PlaneWaveData.__init__()
        """
        # 不要实际使用PlaneWaveData.__init__()原样
        raise NotImplementedError

        # 我们提供以下作为__init__()方法的可视化示例
        nangles, nchans, nsamps = 2, 3, 4  # 示例：2个角度，3个通道，4个采样点
        
        # 初始化子类*必须*填充的参数
        self.idata = np.zeros((nangles, nchans, nsamps), dtype="float32")     # I数据(实部)
        self.qdata = np.zeros((nangles, nchans, nsamps), dtype="float32")     # Q数据(虚部)
        self.angles = np.zeros((nangles,), dtype="float32")                   # 发射角度[弧度]
        self.ele_pos = np.zeros((nchans, 3), dtype="float32")                 # 传感器位置[x,y,z]
        self.fc = 5e6          # 中心频率 5MHz
        self.fs = 20e6         # 采样频率 20MHz
        self.fdemod = 0        # 解调频率 (0表示未解调)
        self.c = 1540          # 声速 1540 m/s (组织中典型值)
        self.time_zero = np.zeros((nangles,), dtype="float32")                # 每个角度的时间零点

    def validate(self):
        """ 
        检查确保所有信息已加载且有效
        
        验证内容：
        1. 数据维度匹配
        2. 频率参数合理
        3. 声速在医学成像范围内
        4. 时间零点配置正确
        """
        
        # ===== 检查idata、qdata、angles、ele_pos的尺寸 =====
        assert self.idata.shape == self.qdata.shape, "I和Q数据形状必须相同"
        assert self.idata.ndim == self.qdata.ndim == 3, "IQ数据必须是3维数组"
        
        nangles, nchans, nsamps = self.idata.shape  # 获取数据维度
        assert self.angles.ndim == 1 and self.angles.size == nangles, f"角度数组维度错误，应为({nangles},)"
        assert self.ele_pos.ndim == 2 and self.ele_pos.shape == (nchans, 3), f"传感器位置形状错误，应为({nchans}, 3)"
        
        # ===== 检查频率 (期望大于0.1 MHz) =====
        assert self.fc > 1e5, f"中心频率过小: {self.fc/1e6:.1f}MHz，应>0.1MHz"
        assert self.fs > 1e5, f"采样频率过小: {self.fs/1e6:.1f}MHz，应>0.1MHz"
        assert self.fdemod > 1e5 or self.fdemod == 0, f"解调频率错误: {self.fdemod/1e6:.1f}MHz"
        
        # ===== 检查声速 (医学成像应在1000-2000之间) =====
        assert 1000 <= self.c <= 2000, f"声速异常: {self.c}m/s，医学成像应在1000-2000m/s"
        
        # ===== 检查每次发射都提供了单独的时间零点 =====
        assert self.time_zero.ndim == 1 and self.time_zero.size == nangles, f"时间零点数量错误，应为{nangles}个"


class PICMUSData(PlaneWaveData):
    """ 
    PICMUS数据加载器 - 演示如何使用PlaneWaveData加载PICMUS数据
    
    PICMUSData是PlaneWaveData的子类，用于加载2016年PICMUS挑战赛的数据
    (https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016)
    
    PICMUSData重新实现了PlaneWaveData的__init__()函数。
    
    文件路径格式：
    {database_path}/{acq}/{target}/{target}_{acq[:4]}_dataset_{dtype}.hdf5
    例如：./datasets/simulation/resolution_distorsion/resolution_distorsion_simu_dataset_iq.hdf5
    """

    def __init__(self, database_path, acq, target, dtype):
        """ 
        加载PICMUS数据集作为PlaneWaveData对象
        
        参数：
            database_path: 数据库路径，如"./datasets"
            acq: 采集类型，"simulation"(仿真)或"experiments"(实验)
            target: 目标类型
                   "contrast_speckle" - 对比度斑点
                   "resolution_distorsion" - 分辨率失真  
                   "carotid_cross" - 颈动脉横截面
                   "carotid_long" - 颈动脉纵截面
            dtype: 数据类型，"rf"(射频)或"iq"(解调)
        """
        
        # ===== 参数验证 =====
        assert any([acq == a for a in ["simulation", "experiments"]]), f"采集类型错误: {acq}"
        assert any([target == t for t in ["contrast_speckle", "resolution_distorsion","carotid_cross","carotid_long"]]), f"目标类型错误: {target}"
        assert any([dtype == d for d in ["rf", "iq"]]), f"数据类型错误: {dtype}"

        # ===== 构建文件路径 =====
        # 数据文件路径：包含IQ或RF数据
        fname = "%s/%s/%s/%s_%s_dataset_%s.hdf5" % (
            database_path,    # ./datasets
            acq,             # simulation 或 experiments  
            target,          # resolution_distorsion 等
            target,          # 重复目标名
            acq[:4],         # simu 或 expe (前4个字符)
            dtype,           # iq 或 rf
        )
        # 例如：./datasets/simulation/resolution_distorsion/resolution_distorsion_simu_dataset_iq.hdf5
        
        # 扫描参数文件路径：包含几何信息
        fname1 = "%s/%s/%s/%s_%s_scan.hdf5" % (
            database_path, acq, target, target, acq[:4]
        )
        # 例如：./datasets/simulation/resolution_distorsion/resolution_distorsion_simu_scan.hdf5

        # ===== 加载HDF5数据文件 =====
        print(f"   加载扫描参数: {fname1}")
        f1 = h5py.File(fname1, "r")["US"]["US_DATASET0000"]  # 扫描参数
        
        print(f"   加载数据文件: {fname}")
        f = h5py.File(fname, "r")["US"]["US_DATASET0000"]    # 实际数据

        # ===== 读取数据 =====
        # IQ数据 (核心超声数据)
        self.idata = np.array(f["data"]["real"], dtype="float32")  # I分量(实部)
        self.qdata = np.array(f["data"]["imag"], dtype="float32")  # Q分量(虚部)
        print(f"   IQ数据形状: {self.idata.shape}")
        
        # 成像坐标轴
        self.x_axis = np.array(f1["x_axis"], dtype="float32")      # 横向坐标轴
        self.z_axis = np.array(f1["z_axis"], dtype="float32")      # 纵向坐标轴
        
        # 采集参数
        self.angles = np.array(f["angles"])                        # 发射角度[弧度]
        self.fc = 5208000.0  # 固定中心频率 5.208MHz (PICMUS标准)
        self.fs = np.array(f["sampling_frequency"]).item()         # 采样频率
        self.c = np.array(f["sound_speed"]).item()                 # 声速
        self.time_zero = np.array(f["initial_time"])               # 初始时间
        self.ele_pos = np.array(f["probe_geometry"]).T             # 探头几何(转置)
        
        # 设置解调频率
        self.fdemod = self.fc if dtype == "iq" else 0              # IQ数据已解调，RF数据未解调
        
        print(f"   发射角度数: {len(self.angles)}")
        print(f"   中心频率: {self.fc/1e6:.2f} MHz")
        print(f"   采样频率: {self.fs/1e6:.2f} MHz")
        print(f"   声速: {self.c} m/s")

        # ===== 处理RF数据 =====
        if dtype == "rf":
            print("   RF数据检测到，执行希尔伯特变换获取虚部...")
            # 如果数据是RF，使用希尔伯特变换获取虚部分量
            iqdata = hilbert(self.idata, axis=-1)  # 沿最后一个轴(时间轴)做希尔伯特变换
            self.qdata = np.imag(iqdata)           # 提取虚部作为Q分量

        # ===== 处理时间零点 =====
        # 确保time_zero是大小为[nangles]的数组
        if self.time_zero.size == 1:
            # 如果只有一个时间零点，复制给所有角度
            self.time_zero = np.ones_like(self.angles) * self.time_zero
            print("   时间零点扩展到所有角度")

        # ===== 验证数据完整性 =====
        print("   验证数据完整性...")
        super().validate()  # 调用父类的validate方法
        print("   ✅ PICMUS数据加载完成并验证通过")


# ===== 使用示例 =====
"""
使用方法：

# 加载仿真分辨率失真IQ数据
data = PICMUSData("./datasets", "simulation", "resolution_distorsion", "iq")

# 访问数据
print(f"IQ数据形状: {data.idata.shape}")  # (角度数, 通道数, 采样点数)
print(f"发射角度: {data.angles}")           # 弧度
print(f"传感器位置: {data.ele_pos}")         # 米
print(f"中心频率: {data.fc/1e6:.1f} MHz")   # MHz
print(f"声速: {data.c} m/s")               # m/s

AUGAN中的使用：
1. 加载PICMUS数据：P = PICMUSData(...)
2. 创建DAS网络：das = DAS_PW(P, grid, angle_list)
3. 重建图像：img = mk_img(das, (P.idata, P.qdata))
"""