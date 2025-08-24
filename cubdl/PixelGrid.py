#!/usr/bin/env python3
"""
PixelGrid详细注释版本
原文件：PixelGrid.py
作者：Dongwoon Hyun (dongwoon.hyun@stanford.edu)
创建日期：2020-04-03
作用：创建超声成像的像素网格，定义重建图像的空间坐标系统
"""

import numpy as np

eps = 1e-10  # 极小值，避免浮点数精度问题


def make_pixel_grid(xlims, zlims, dx, dz):
    """ 
    根据输入参数生成像素网格
    
    这个函数是超声成像的核心基础设施，定义了图像重建的空间坐标系统。
    在DAS波束成形中，算法需要知道要在哪些空间位置重建像素，
    这个函数就是创建这些目标位置的坐标网格。
    
    参数：
    xlims: 横向范围 [xmin, xmax]，单位：米
           - 定义成像区域的左右边界
           - 对应超声阵列的横向方向（阵元排列方向）
           - 例如：[-0.02, 0.02] 表示左右各2cm的成像宽度
           
    zlims: 纵向范围 [zmin, zmax]，单位：米  
           - 定义成像区域的深度范围
           - z=0通常对应阵列表面，正z方向向下
           - 例如：[0.005, 0.055] 表示0.5cm到5.5cm深度
           
    dx: 横向像素间距，单位：米
        - 控制横向分辨率，越小分辨率越高但计算量越大
        - 典型值：50-200微米
        
    dz: 纵向像素间距，单位：米
        - 控制纵向分辨率，通常等于dx保持方形像素
        - 典型值：50-200微米
    
    返回：
    grid: 像素网格数组，形状为 [nrows, ncols, 3]
          - nrows = (zlims[1] - zlims[0]) / dz + 1，图像高度
          - ncols = (xlims[1] - xlims[0]) / dx + 1，图像宽度  
          - 最后一维包含每个像素的(x, y, z)坐标
          - y坐标始终为0（2D成像，无侧向维度）
    
    使用示例：
    >>> xlims = [-0.02, 0.02]    # 左右各2cm
    >>> zlims = [0.005, 0.055]   # 深度0.5cm到5.5cm
    >>> dx = dz = 100e-6         # 100微米像素
    >>> grid = make_pixel_grid(xlims, zlims, dx, dz)
    >>> print(f"图像尺寸: {grid.shape}")
    图像尺寸: (501, 401, 3)  # 501行×401列×3坐标
    """
    
    # ===== 创建一维坐标轴 =====
    # 横向坐标轴：从xlims[0]到xlims[1]，步长为dx
    # 加上eps避免浮点数精度导致边界点丢失
    x = np.arange(xlims[0], xlims[1] + eps, dx)
    
    # 纵向坐标轴：从zlims[0]到zlims[1]，步长为dz  
    z = np.arange(zlims[0], zlims[1] + eps, dz)
    
    # ===== 创建2D网格 =====
    # 使用meshgrid创建2D坐标网格，indexing="xy"表示x对应列，z对应行
    # xx: 每个网格点的x坐标，形状为 [nrows, ncols]
    # zz: 每个网格点的z坐标，形状为 [nrows, ncols]
    xx, zz = np.meshgrid(x, z, indexing="xy")
    
    # ===== 添加y维度 =====
    # 超声成像通常是2D的，y坐标设为0
    # yy: 全零数组，与xx、zz形状相同
    yy = 0 * xx
    
    # ===== 组合为3D坐标网格 =====
    # 将(x,y,z)坐标沿最后一个维度堆叠
    # 最终形状：[nrows, ncols, 3]
    # grid[i, j, :] = [x坐标, y坐标, z坐标] 对应第i行第j列像素的3D位置
    grid = np.stack((xx, yy, zz), axis=-1)
    
    return grid


def make_foctx_grid(rlims, dr, oris, dirs):
    """ 
    为聚焦发射创建像素网格
    
    这个函数专门用于聚焦发射成像，创建极坐标形式的网格。
    与平面波成像的直角坐标网格不同，聚焦发射更适合用极坐标描述。
    
    参数：
    rlims: 径向范围 [rmin, rmax]，单位：米
           - 定义从发射焦点开始的径向距离范围
           - 类似于平面波成像中的深度范围
           
    dr: 径向像素间距，单位：米
        - 控制径向分辨率
        
    oris: 发射起始点，形状为 [nfoci, 3]
          - 每个聚焦发射的空间起始位置
          - 通常对应不同的聚焦深度或角度
          
    dirs: 发射方向，形状为 [nfoci, 2]  
          - 每个聚焦发射的方向角度 [方位角, 俯仰角]
          - 只使用方位角进行计算（2D成像）
    
    返回：
    grid: 聚焦发射网格，形状为 [nfoci, nr, 3]
          - nfoci: 聚焦发射次数
          - nr: 径向采样点数
          - 最后一维：每个点的(x, y, z)坐标
    
    物理含义：
    对于每个聚焦发射，创建一条从起始点沿发射方向延伸的射线，
    射线上均匀分布像素点用于图像重建。这种方式适合聚焦发射的
    扇形或扇扇形成像几何。
    """
    
    # ===== 创建极坐标网格 =====
    # 径向坐标：从rlims[0]到rlims[1]，步长为dr
    r = np.arange(rlims[0], rlims[1] + eps, dr)  # 径向距离rho
    
    # 角向坐标：使用发射方向的方位角theta（忽略俯仰角phi）
    t = dirs[:, 0]  # 提取方位角
    
    # 创建2D极坐标网格
    # rr: 每个网格点的径向距离，形状为 [nfoci, nr]
    # tt: 每个网格点的角度，形状为 [nfoci, nr]
    rr, tt = np.meshgrid(r, t, indexing="xy")

    # ===== 极坐标转直角坐标 =====
    # 极坐标(r, θ)转换为直角坐标(x, z)：
    # x = r * sin(θ) + x_origin
    # z = r * cos(θ) + z_origin
    
    # 横向坐标：径向距离×sin(角度) + 起始点x坐标
    xx = rr * np.sin(tt) + oris[:, [0]]
    
    # 纵向坐标：径向距离×cos(角度) + 起始点z坐标  
    zz = rr * np.cos(tt) + oris[:, [2]]
    
    # 侧向坐标：设为0（2D成像）
    yy = 0 * xx
    
    # ===== 组合为3D坐标网格 =====
    grid = np.stack((xx, yy, zz), axis=-1)
    
    return grid


# ===== 使用说明和对比 =====
"""
像素网格在超声成像中的作用：

1. 空间参考系统：
   - 定义图像重建的目标位置
   - 统一数据处理的坐标系统
   - 连接物理空间和图像空间

2. 分辨率控制：
   - dx, dz越小 -> 分辨率越高，计算量越大
   - dx, dz越大 -> 分辨率越低，计算速度快
   - 典型权衡：50-200微米像素间距

3. 成像区域定义：
   - xlims控制横向视野（阵列宽度相关）
   - zlims控制成像深度（穿透能力相关）

4. 两种网格类型：
   - make_pixel_grid: 直角坐标，适合平面波成像
   - make_foctx_grid: 极坐标，适合聚焦发射成像

在AUGAN项目中的使用：
- 主要使用make_pixel_grid创建直角坐标网格
- 用于平面波DAS重建和评估指标计算
- 确保输入图像和目标图像使用相同的空间坐标系

使用示例：
```python
# AUGAN中的典型用法
xlims = [-0.0192, 0.0192]     # ±1.92cm横向范围
zlims = [0.005, 0.055]        # 0.5cm到5.5cm深度
dx = dz = 100e-6              # 100微米像素
grid = make_pixel_grid(xlims, zlims, dx, dz)

# 创建DAS网络
das = DAS_PW(plane_wave_data, grid, ang_list)

# 重建图像
img_i, img_q = das((idata, qdata))
img = np.sqrt(img_i**2 + img_q**2)  # 图像强度
```

坐标系约定：
- x轴：横向方向，阵元排列方向，左负右正
- y轴：侧向方向，2D成像中通常为0
- z轴：纵向方向，深度方向，向下为正
- 原点：通常在阵列中心表面
"""