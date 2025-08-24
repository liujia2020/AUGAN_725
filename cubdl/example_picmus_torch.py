#!/usr/bin/env python3
"""
PICMUS超声数据处理详细注释版
原文件：example_picmus_torch.py
作用：加载PICMUS数据集，执行DAS重建，生成超声图像
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from cubdl.das_torch import DAS_PW
from cubdl.PlaneWaveData import PICMUSData
from cubdl.PixelGrid import make_pixel_grid

eps = 2.2204e-16  # 很小的数，避免log(0)

# ===== 用户自定义参数 =====
phase            = 1     # 1=生成train数据；2=生成test数据
acquisition_type = 2     # 1=simulation仿真；2=experiments实验
phantom_type     = 4     # 1=分辨率 2=对比度 3=颈动脉横截面 4=颈动脉纵截面
data_type        = 1     # 1=iq解调数据；2=rf射频数据


def load_datasets(acq, target, dtype):
    """
    加载PICMUS数据集
    
    参数:
        acq: "simulation"(仿真) 或 "experiments"(实验)
        target: 数据集类型
               "resolution_distorsion" - 分辨率失真
               "contrast_speckle" - 对比度斑点
               "carotid_cross" - 颈动脉横截面
               "carotid_long" - 颈动脉纵截面
        dtype: "iq"(解调数据) 或 "rf"(射频数据)
    
    返回:
        P: PICMUSData对象，包含超声数据的所有信息
    """
    database_path = "./datasets"  # 数据集路径
    
    # 参数检查
    assert acq == "simulation" or acq == "experiments" 
    assert target in ["resolution_distorsion", "contrast_speckle", "carotid_cross", "carotid_long"]
    assert dtype == "iq" or dtype == "rf"
    
    # 加载数据
    P = PICMUSData(database_path, acq, target, dtype)
    
    return P


def create_network(P, angle_list):
    """
    创建DAS重建网络
    
    参数:
        P: PICMUSData对象，包含超声数据
        angle_list: 要使用的角度列表，例如[1,2,3]或[37]
    
    返回:
        das: DAS重建网络
        iqdata: IQ数据元组 (idata, qdata)
        xlims: 横向成像范围 [最小x, 最大x]
        zlims: 纵向成像范围 [最小z, 最大z]
    """
    
    # ===== 定义成像区域 =====
    # 横向范围：从第一个传感器到最后一个传感器
    xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
    # 纵向范围：从5mm到55mm深度
    zlims = [5e-3, 55e-3]  # 单位：米
    
    # ===== 计算像素间距 =====
    wvln = P.c / P.fc  # 波长 = 声速 / 中心频率
    dx = wvln / 3      # 横向像素间距 = 波长/3
    dz = dx            # 纵向像素间距 = 横向像素间距 (方形像素)
    
    # ===== 创建像素网格 =====
    grid = make_pixel_grid(xlims, zlims, dx, dz)
    fnum = 1  # F数（焦点参数）
    
    # ===== 创建DAS网络 =====
    # DAS_PW: 平面波延迟求和重建网络
    das = DAS_PW(P, grid, angle_list)

    # ===== 准备IQ数据 =====
    # 将I和Q分量打包成元组
    iqdata = (P.idata, P.qdata)
    
    return das, iqdata, xlims, zlims


def mk_img(dasN, iqdata):
    """
    使用DAS网络生成超声图像
    
    参数:
        dasN: DAS重建网络
        iqdata: IQ数据元组 (idata, qdata)
    
    返回:
        bimgN: 重建的超声图像 (dB格式)
    """
    
    # ===== 执行DAS重建 =====
    idasN, qdasN = dasN.forward(iqdata)  # 前向传播得到重建的IQ数据
    
    # ===== 转换到CPU并转为numpy =====
    idasN, qdasN = idasN.detach().cpu().numpy(), qdasN.detach().cpu().numpy()
    # detach(): 分离计算图，不计算梯度
    # cpu(): 从GPU移到CPU
    # numpy(): 转换为numpy数组
    
    # ===== 归一化 =====
    idasN = idasN / np.max(idasN)  # I分量归一化
    qdasN = qdasN / np.max(qdasN)  # Q分量归一化
    
    # ===== 计算复数信号幅度 =====
    iqN = idasN + 1j * qdasN  # 组合成复数: I + jQ
    iqN = iqN + eps           # 加小数避免log(0)
    
    # ===== 对数压缩 (转换为dB) =====
    bimgN = 20 * np.log10(np.abs(iqN))  # 20*log10(幅度) = dB
    # bimgN -= np.amax(bimgN)  # (可选)归一化到最大值为0dB
    
    return bimgN


def dispaly_img(bimg1, bimg2, bimg3, xlims, zlims, angle_list, epoch, phase, name):
    """
    显示三张超声图像对比
    
    参数:
        bimg1: 第一张图像 (通常是输入/单角度)
        bimg2: 第二张图像 (通常是生成/重建)
        bimg3: 第三张图像 (通常是目标/多角度)
        xlims, zlims: 成像范围
        angle_list: 使用的角度列表
        epoch: 当前epoch数
        phase: 训练阶段 ('train' or 'test')
        name: 实验名称
    """
    
    # ===== 计算显示范围 =====
    extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
    extent2 = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
    extent3 = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
    # 转换为毫米单位显示

    # ===== 显示第一张图像 =====
    plt.subplot(131)  # 1行3列，第1个
    plt.imshow(bimg1, vmin=-60, vmax=0, cmap="gray", extent=extent, origin="upper")
    if phase == 'train':
        plt.title("%d epochs LR image" % epoch, fontsize=10)
    elif phase == 'test':
        plt.title("%d test LR image" % epoch, fontsize=10)

    # ===== 显示第二张图像 =====
    plt.subplot(132)  # 1行3列，第2个
    plt.imshow(bimg2, vmin=-60, vmax=0, cmap="gray", extent=extent2, origin="upper")
    if phase == 'train':
        plt.title("%d epochs generated image" % epoch, fontsize=10)
    elif phase == 'test':
        plt.title("%d test generated image" % epoch, fontsize=10)

    # ===== 显示第三张图像 =====
    plt.subplot(133)  # 1行3列，第3个
    plt.imshow(bimg3, vmin=-60, vmax=0, cmap="gray", extent=extent3, origin="upper")
    if phase == 'train':
        plt.title("%d epochs HR image" % epoch, fontsize=10)
    elif phase == 'test':
        plt.title("%d test HR image" % epoch, fontsize=10)

    # ===== 保存图像 =====
    plt.savefig('./images/%s/%s/%d_%s.png' % (name, phase, epoch, phase), 
                bbox_inches='tight', dpi=150)
    plt.close()  # 关闭图像释放内存


# ===== 主要函数总结 =====
"""
核心函数流程：
1. load_datasets() - 加载PICMUS数据集
2. create_network() - 创建DAS重建网络和成像参数
3. mk_img() - 执行DAS重建生成超声图像
4. dispaly_img() - 显示和保存图像对比

AUGAN训练中的使用：
- 单角度重建：create_network(data, [单个角度]) → mk_img() → 输入图像A
- 多角度重建：create_network(data, [所有角度]) → mk_img() → 目标图像B
- 训练对：(A, B) 用于训练Pix2Pix模型
"""