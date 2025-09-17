#!/usr/bin/env python3
"""
utils/metrics详细注释版本
原文件：utils/metrics.py
作者：Tang Jiahua
创建日期：2021-06-26
作用：AUGAN项目的完整图像评估系统，提供超声图像质量评估的综合指标
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from cubdl.PixelGrid import make_pixel_grid
# import pytorch_ssim
from utils.pytorch_ssim import ssim as pytorch_ssim_ssim
import torch
from sklearn import metrics


def contrast(img1, img2):
    """
    计算对比度比值
    
    参数:
        img1, img2: 两个图像区域
        
    返回:
        对比度比值 = img1均值 / img2均值
        
    应用:
        评估图像中不同区域的对比度差异
    """
    return img1.mean() / img2.mean()


def cnr(img1, img2):
    """
    计算对比噪声比 (Contrast-to-Noise Ratio)
    
    参数:
        img1, img2: 两个图像区域
        
    返回:
        CNR = |均值差| / √(方差和)
        
    物理含义:
        衡量信号对比度相对于噪声的强弱
        值越大表示对比度越好，噪声影响越小
    """
    return np.abs(img1.mean() - img2.mean()) / np.sqrt(img1.var() + img2.var())


def gcnr(img1, img2):
    """
    计算广义对比噪声比 (Generalized Contrast-to-Noise Ratio)
    
    参数:
        img1, img2: 两个图像区域
        
    返回:
        GCNR = 1 - 两个概率分布的重叠度
        
    物理含义:
        基于直方图分布的对比度测量
        0表示完全重叠（无对比），1表示完全分离（最佳对比）
        相比传统CNR，对非高斯噪声更鲁棒
    """
    # 合并两个图像数据计算共同的直方图区间
    a = np.concatenate((img1, img2))
    _, bins = np.histogram(a, bins=256)
    
    # 计算各自的概率密度分布
    f, _ = np.histogram(img1, bins=bins, density=True)
    g, _ = np.histogram(img2, bins=bins, density=True)
    
    # 归一化为概率分布
    f /= f.sum()
    g /= g.sum()
    
    # 计算分布间的不重叠度
    return 1 - np.sum(np.minimum(f, g))


def MI(img1, img2):
    """
    计算两个图像间的归一化互信息 (Normalized Mutual Information)
    
    参数:
        img1, img2: 两个输入图像
        
    返回:
        归一化互信息值 [0, 1]
        
    物理含义:
        测量两个图像间的信息相似性
        1表示完全相关，0表示完全独立
        
    改进:
        - 将连续值量化为256个离散级别避免sklearn警告
        - 使用warnings上下文管理器抑制警告
    """
    image1 = np.squeeze(img1)
    image2 = np.squeeze(img2)
    
    # ===== 图像值量化处理 =====
    # 将连续值量化为离散值（256个级别）
    # 这样可以避免sklearn的警告，同时保持合理的精度
    image1_quantized = np.round((image1 - image1.min()) / (image1.max() - image1.min()) * 255).astype(int)
    image2_quantized = np.round((image2 - image2.min()) / (image2.max() - image2.min()) * 255).astype(int)
    
    # ===== 计算归一化互信息 =====
    # 抑制sklearn的警告
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_NMI = metrics.normalized_mutual_info_score(
            image1_quantized.flatten(), 
            image2_quantized.flatten()
        )
    
    return result_NMI


def shan_entropy(c):
    """
    计算香农熵 (Shannon Entropy)
    
    参数:
        c: 概率分布或计数数组
        
    返回:
        香农熵值
        
    物理含义:
        信息熵，衡量信息的不确定性
        值越大表示信息越混乱/不确定
    """
    # 归一化为概率分布
    c_normalized = c / float(np.sum(c))
    # 移除零值避免log(0)
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    # 计算香农熵: H = -∑p*log2(p)
    H = -sum(np.log2(c_normalized))
    return H


def res_FWHM(img):
    """
    计算半高宽分辨率 (Full Width at Half Maximum)
    
    注意: 当前版本未实现
    
    用途:
        测量超声图像中点目标的空间分辨率
        通过测量响应峰值一半处的宽度来评估分辨率
    """
    # TODO: Write FWHM code
    raise NotImplementedError


def speckle_res(img):
    """
    计算散斑边缘扩散函数分辨率
    
    注意: 当前版本未实现
    
    用途:
        评估超声图像中散斑模式的分辨率特性
        基于边缘扩散函数进行分辨率估计
    """
    # TODO: Write speckle edge-spread function resolution code
    raise NotImplementedError


def snr(img):
    """
    计算信噪比 (Signal-to-Noise Ratio)
    
    参数:
        img: 输入图像
        
    返回:
        SNR = 均值 / 标准差
        
    物理含义:
        信号强度相对于噪声的比值
        值越大表示图像质量越好
    """
    return img.mean() / img.std()


def l1loss(img1, img2):
    """
    计算L1损失 (平均绝对误差)
    
    参数:
        img1, img2: 两个图像
        
    返回:
        L1 = mean(|img1 - img2|)
        
    应用:
        评估图像重建的像素级误差
        对异常值相对鲁棒
    """
    return np.abs(img1 - img2).mean()


def l2loss(img1, img2):
    """
    计算L2损失 (均方根误差)
    
    参数:
        img1, img2: 两个图像
        
    返回:
        L2 = √(mean((img1 - img2)²))
        
    应用:
        评估图像重建的像素级误差
        对大误差更加敏感
    """
    return np.sqrt(((img1 - img2) ** 2).mean())


def psnr(img1, img2):
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio)
    
    参数:
        img1, img2: 两个图像
        
    返回:
        PSNR = 20 * log10(动态范围 / L2误差)
        
    物理含义:
        图像质量的客观评价指标
        值越大表示图像失真越小
        通常用于评估图像压缩和重建质量
    """
    dynamic_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    return 20 * np.log10(dynamic_range / l2loss(img1, img2))


def ncc(img1, img2):
    """
    计算归一化交叉相关 (Normalized Cross Correlation)
    
    参数:
        img1, img2: 两个图像
        
    返回:
        NCC: 归一化相关系数 [-1, 1]
        
    物理含义:
        衡量两个图像的形状相似性
        1表示完全相关，0表示不相关，-1表示完全负相关
    """
    # 计算去均值后的归一化相关
    return ((img1-img1.mean()) * (img2-img2.mean())).sum() / \
           np.sqrt(((img1-img1.mean()) ** 2).sum() * ((img2-img2.mean()) ** 2).sum())


def Compute_6dB_Resolution(x_axis, y_signal):
    """
    计算6dB分辨率
    
    参数:
        x_axis: 空间轴坐标
        y_signal: 对应的信号强度
        
    返回:
        6dB分辨率: 信号峰值下降6dB处的宽度
        
    物理含义:
        超声成像中常用的分辨率指标
        测量信号强度下降到峰值一半(-6dB)时的空间宽度
        
    算法步骤:
        1. 对信号进行插值增加采样密度
        2. 找到峰值位置
        3. 找到峰值-6dB处的两个边界点
        4. 计算两点间的距离
    """
    coeff = 10  # 插值系数，增加10倍采样密度
    nb_sample = np.size(x_axis)
    nb_interp = nb_sample * coeff
    x_interp = np.linspace(x_axis[0], x_axis[nb_sample-1], nb_interp)
    
    # ===== 数据预处理 =====
    # 确保 y_signal 是1D数组
    if y_signal.ndim > 1:
        y_signal = y_signal.flatten()
    
    # 如果长度不匹配，取最短的长度
    min_len = min(len(x_axis), len(y_signal))
    x_axis = x_axis[:min_len]
    y_signal = y_signal[:min_len]
    
    # 重新计算插值参数
    nb_sample = len(x_axis)
    nb_interp = nb_sample * coeff
    x_interp = np.linspace(x_axis[0], x_axis[nb_sample-1], nb_interp)
    
    # ===== 信号插值 =====
    try:
        y_interp = np.interp(x_interp, x_axis, y_signal)
    except Exception as e:
        # 如果插值失败，返回默认值
        return 0.0

    # ===== 计算6dB分辨率 =====
    try:
        # 找到峰值-6dB处的所有点
        ind = np.where(y_interp >= (np.max(y_interp) - 6))
        if len(ind[0]) == 0:
            return 0.0
        
        # 获取边界点索引
        idx1 = np.min(ind)
        idx2 = np.max(ind)
        
        # 计算分辨率（两个边界点间的距离）
        res = x_interp[idx2] - x_interp[idx1]
        return res
    except Exception as e:
        # 如果最终计算失败，返回默认值
        return 0.0


class image_evaluation():
    """
    图像评估类 - AUGAN项目的综合图像质量评估系统
    
    功能:
        为不同类型的超声图像测试提供专业的评估指标
        支持点目标、体模目标、实验和体内数据的评估
        
    测试类型:
        test_type = 1: 仿真点目标
        test_type = 2: 仿真体模目标  
        test_type = 3: 实验点目标
        test_type = 4: 实验体模目标
        test_type = 5: 体内目标
        
    评估指标:
        - FWHM: 半高宽分辨率
        - CR: 对比度
        - CNR: 对比噪声比
        - sSNR: 散斑信噪比
        - GCNR: 广义对比噪声比
        - PSNR: 峰值信噪比
        - NCC: 归一化交叉相关
        - L1/L2 Loss: 重建误差
        - SSIM: 结构相似性指数
        - MI: 互信息
    """
    
    def __init__(self):
        """
        初始化评估系统
        
        创建结果字典存储各种评估指标
        初始化平均分数变量
        """
        # 结果存储字典
        self.result = {
            'CR': [],      # 对比度
            'CNR': [],     # 对比噪声比
            'sSNR': [],    # 散斑信噪比
            'GCNR': [],    # 广义对比噪声比
            'PSNR': [],    # 峰值信噪比
            'NCC': [],     # 归一化交叉相关
            'L1loss': [],  # L1损失
            'L2loss': [],  # L2损失
            'FWHM': [],    # 半高宽分辨率
            'SSIM': [],    # 结构相似性
            'MI': []       # 互信息
        }
        
        # 平均分数变量
        self.average_score_FWHM = 0
        self.average_score_GCNR = 0
        self.average_score_sSNR = 0
        self.average_score_CNR = 0
        self.average_score_CR = 0

    def evaluate(self, img1, img2, opt, plane_wave_data, test_type, i):
        """
        执行综合图像评估
        
        参数:
            img1: 输入图像（通常是GAN生成的结果）
            img2: 目标图像（通常是真实高质量图像）
            opt: 训练/测试选项
            plane_wave_data: 平面波数据对象，包含几何信息
            test_type: 测试类型 (1-5)
            i: 当前测试图像索引
            
        评估流程:
            1. 图像预处理（归一化到峰值）
            2. 根据测试类型设置ROI参数
            3. 计算专业超声评估指标
            4. 计算通用图像质量指标
            5. 存储结果到结果字典
        """
        
        # ===== 图像预处理 =====
        # 归一化到峰值（超声成像的标准显示方式）
        img1 -= np.max(img1)
        img2 -= np.max(img2)

        # ===== 体模目标测试配置 =====
        if test_type == 2:  # 仿真体模目标
            # 体模中的圆形目标参数（9个目标）
            self.occlusionDiameter = np.array([0.008, 0.008, 0.008, 0.008, 0.008, 
                                             0.008, 0.008, 0.008, 0.008])
            self.r = 0.004          # 目标半径
            self.rin = self.r - 6.2407e-4    # 内部ROI半径
            self.rout1 = self.r + 6.2407e-4  # 外部ROI内半径
            self.rout2 = 1.2 * np.sqrt(self.rin*self.rin + self.rout1*self.rout1)  # 外部ROI外半径
            
            # 9个目标的中心位置（3×3网格）
            self.xcenter = [0, 0, 0, -0.012, -0.012, -0.012, 0.012, 0.012, 0.012]
            self.zcenter = [0.018, 0.03, 0.042, 0.018, 0.03, 0.042, 0.018, 0.03, 0.042]

        if test_type == 4:  # 实验体模目标
            # 实验数据中的目标参数（2个目标）
            self.occlusionDiameter = np.array([0.0045, 0.0045])
            self.r = 0.0022
            self.rin = self.r - 6.2407e-4
            self.rout1 = self.r + 6.2407e-4
            self.rout2 = 1.2 * np.sqrt(self.rin * self.rin + self.rout1 * self.rout1)
            
            # 根据图像索引设置不同的目标位置
            if i <= 30:
                self.xcenter = [-1.0e-04, -1.0e-04]
                self.zcenter = [0.0149, 0.0428]
            elif i > 30 and i <= 45:
                self.xcenter = [1.0e-04, 1.0e-04]
                self.zcenter = [0.0149, 0.0428]
            else:  # i > 45
                self.xcenter = [-1.0e-04, -1.0e-04]
                self.zcenter = [0.0172, 0.0451]

        # ===== 创建像素网格（用于ROI定义） =====
        xlims = [plane_wave_data.ele_pos[0, 0], plane_wave_data.ele_pos[-1, 0]]
        zlims = [5e-3, 55e-3]  # 5-55mm深度范围
        wvln = plane_wave_data.c / plane_wave_data.fc  # 波长
        dx = wvln / 3  # 像素间距（波长的1/3）
        dz = dx        # 正方形像素
        grid = make_pixel_grid(xlims, zlims, dx, dz)
        self.x_matrix = grid[:, :, 0]  # x坐标矩阵
        self.z_matrix = grid[:, :, 2]  # z坐标矩阵

        image = img1

        # ===== 点目标测试：FWHM分辨率评估 =====
        if test_type == 1 or test_type == 3:  # 点目标测试
            maskROI = np.zeros((508, 387))
            
            # 根据测试类型和图像索引定义点目标位置
            if test_type == 1 and i <= 45:  # 仿真点目标（前45张）
                sca = np.array([
                    [0, 0, 0.01], [0, 0, 0.015], [0, 0, 0.02], [0, 0, 0.025], [0, 0, 0.03],
                    [0, 0, 0.035], [0, 0, 0.04], [0, 0, 0.045],
                    [-0.015, 0, 0.02], [-0.01, 0, 0.02], [-0.005, 0, 0.02], 
                    [0.005, 0, 0.02], [0.01, 0, 0.02], [0.015, 0, 0.02],
                    [-0.015, 0, 0.04], [-0.01, 0, 0.04], [-0.005, 0, 0.04], 
                    [0.005, 0, 0.04], [0.01, 0, 0.04], [0.015, 0, 0.04]
                ])
            elif test_type == 1 and i > 45:  # 仿真点目标（后面的）
                sca = np.array([
                    [0, 0, 0.015], [0, 0, 0.02], [0, 0, 0.025], [0, 0, 0.03], [0, 0, 0.035],
                    [0, 0, 0.04], [0, 0, 0.045], [0, 0, 0.05],
                    [-0.015, 0, 0.02], [-0.01, 0, 0.02], [-0.005, 0, 0.02], 
                    [0.005, 0, 0.02], [0.01, 0, 0.02], [0.015, 0, 0.02],
                    [-0.015, 0, 0.04], [-0.01, 0, 0.04], [-0.005, 0, 0.04], 
                    [0.005, 0, 0.04], [0.01, 0, 0.04], [0.015, 0, 0.04]
                ])
            elif test_type == 3:  # 实验点目标
                if i <= 30:
                    sca = np.array([
                        [-0.0005, 0, 0.0096], [-0.0004, 0, 0.0187], [-0.0004, 0, 0.028],
                        [-0.0002, 0, 0.0376], [-0.0001, 0, 0.047],
                        [-0.0105, 0, 0.0375], [0.0098, 0, 0.0376]
                    ])
                elif i > 30 and i <= 45:
                    sca = np.array([
                        [0.0005, 0, 0.0096], [0.0004, 0, 0.0187], [0.0004, 0, 0.028],
                        [0.0002, 0, 0.0376], [0.0001, 0, 0.047],
                        [0.0105, 0, 0.0375], [-0.0098, 0, 0.0376]
                    ])
                else:  # i > 45
                    sca = np.array([
                        [-0.0005, 0, 0.0504], [-0.0004, 0, 0.0413], [-0.0004, 0, 0.032],
                        [-0.0002, 0, 0.0224], [-0.0001, 0, 0.013],
                        [-0.0105, 0, 0.0225], [0.0098, 0, 0.0224]
                    ])

            # ===== 为每个点目标创建ROI掩码 =====
            for k in range(sca.shape[0]):
                x = sca[k][0]  # 目标x坐标
                z = sca[k][2]  # 目标z坐标
                # 创建3.6mm×3.6mm的方形ROI
                mask = (k+1) * ((self.x_matrix > (x-0.0018)) & (self.x_matrix < (x+0.0018)) & 
                               (self.z_matrix > (z-0.0018)) & (self.z_matrix < (z+0.0018)))
                maskROI = maskROI + mask

            # ===== 准备图像数据 =====
            patchImg1 = np.zeros((508, 387))
            patchImg1[0:508, 0:384] = image[0][0:508, :]
            patchImg1[:, 384:387] = patchImg1[:, 381:384]  # 边界填充

            # ===== 为每个点目标计算FWHM分辨率 =====
            score1 = np.zeros((sca.shape[0], 2))  # [轴向分辨率, 横向分辨率]

            for k in range(sca.shape[0]):
                # 提取当前目标的图像patch
                patchMask = np.copy(maskROI)
                patchImg = np.copy(patchImg1)
                patchImg[maskROI != (k+1)] = np.min(np.min(min(image)))
                patchMask[maskROI != (k+1)] = 0
                
                # 找到ROI边界
                [idzz, idxx] = np.where(patchMask == (k+1))
                
                # 计算patch的空间范围
                x_lim_patch = np.array([plane_wave_data.x_axis[np.min(idxx)],
                                      plane_wave_data.x_axis[np.max(idxx)]]) * 1e3
                z_lim_patch = np.array([plane_wave_data.z_axis[np.min(idzz)],
                                      plane_wave_data.z_axis[np.max(idzz)]]) * 1e3
                
                # 提取坐标轴
                a = np.arange(np.min(idxx), np.max(idxx)+1)
                x_patch = plane_wave_data.x_axis[a] * 1e3
                b = np.arange(np.min(idzz), np.max(idzz)+1)
                z_patch = plane_wave_data.z_axis[b] * 1e3

                # 找到峰值位置
                [idz, idx] = np.where(patchImg == np.max(np.max(patchImg)))
                
                # 提取横向和轴向剖面
                signalLateral = patchImg[idz, np.min(idxx):(np.max(idxx)+1)]
                signalAxial = patchImg[np.min(idzz):(np.max(idzz)+1), idx]

                # 计算6dB分辨率
                res_axial = Compute_6dB_Resolution(z_patch, signalAxial)
                res_lateral = Compute_6dB_Resolution(x_patch, signalLateral)
                
                score1[k][0] = res_axial
                score1[k][1] = res_lateral

            self.average_score_FWHM = np.mean(score1)

        # ===== 体模目标测试：对比度评估 =====
        if test_type == 2 or test_type == 4:  # 体模目标测试
            score2 = np.zeros(self.occlusionDiameter.shape[0])  # 对比度
            score3 = np.zeros(self.occlusionDiameter.shape[0])  # CNR
            score4 = np.zeros(self.occlusionDiameter.shape[0])  # sSNR
            score5 = np.zeros(self.occlusionDiameter.shape[0])  # GCNR
            
            for k in range(len(self.occlusionDiameter)):
                xc = self.xcenter[k]  # 目标中心x坐标
                zc = self.zcenter[k]  # 目标中心z坐标
                
                # ===== 定义ROI掩码 =====
                # 目标区域掩码（圆形）
                maskOcclusion = (np.power(self.x_matrix-xc, 2) + 
                               np.power(self.z_matrix-zc, 2)) <= (self.r * self.r)
                
                # 内部ROI（目标内部）
                maskInside = (np.power(self.x_matrix-xc, 2) + 
                            np.power(self.z_matrix-zc, 2)) <= (self.rin * self.rin)
                
                # 外部ROI（背景区域，环形）
                a = (np.power(self.x_matrix-xc, 2) + 
                    np.power(self.z_matrix-zc, 2)) >= (self.rout1 * self.rout1)
                b = (np.power(self.x_matrix-xc, 2) + 
                    np.power(self.z_matrix-zc, 2)) <= (self.rout2 * self.rout2)
                maskOutside = a & b

                # ===== 提取ROI内的像素值 =====
                inside = []   # 目标内部像素值
                outside = []  # 背景像素值
                
                for i in range(508):
                    for j in range(384):
                        if maskInside[i][j] == True:
                            inside.append(image[0][i][j])
                        if maskOutside[i][j] == True:
                            outside.append(image[0][i][j])
                
                outside = np.array(outside)
                inside = np.array(inside)

                # ===== 计算对比度相关指标 =====
                # 对比度（绝对均值差）
                CR1 = np.abs(np.mean(inside) - np.mean(outside))
                
                # 对比噪声比
                CNR = np.abs(np.mean(inside) - np.mean(outside)) / \
                      np.sqrt((inside.var() + outside.var()))
                
                # 散斑信噪比
                sSNR = np.abs(np.mean(outside)) / np.std(outside)
                
                # 广义对比噪声比
                GCNR = gcnr(inside, outside)

                # 存储当前目标的结果
                score2[k] = CR1
                score3[k] = CNR
                score4[k] = sSNR
                score5[k] = GCNR
                
            # 计算所有目标的平均分数
            self.average_score_CR = np.mean(score2)
            self.average_score_CNR = np.mean(score3)
            self.average_score_sSNR = np.mean(score4)
            self.average_score_GCNR = np.mean(score5)

        # ===== 计算通用图像质量指标 =====
        self.PSNR = psnr(img1, img2)  # 峰值信噪比
        self.MI = MI(img1, img2)      # 互信息

        # ===== 计算SSIM =====
        # 转换为PyTorch张量进行SSIM计算
        ima1 = torch.from_numpy(img1)
        ima2 = torch.from_numpy(img2)
        ima1 = torch.unsqueeze(ima1, 1)  # 添加通道维度
        ima2 = torch.unsqueeze(ima2, 1)
        # self.SSIM = pytorch_ssim.ssim(ima1, ima2)
        self.SSIM = pytorch_ssim_ssim(ima1, ima2)
        # ===== 计算其他指标 =====
        self.L1Loss = l1loss(img1, img2)  # L1损失
        self.L2Loss = l2loss(img1, img2)  # L2损失
        self.NCC = ncc(img1, img2)        # 归一化交叉相关

        # ===== 存储评估结果 =====
        self.result['FWHM'].append(self.average_score_FWHM)
        self.result['CR'].append(self.average_score_CR)
        self.result['CNR'].append(self.average_score_CNR)
        self.result['sSNR'].append(self.average_score_sSNR)
        self.result['GCNR'].append(self.average_score_GCNR)
        self.result['L1loss'].append(self.L1Loss)
        self.result['L2loss'].append(self.L2Loss)
        self.result['PSNR'].append(self.PSNR)
        self.result['NCC'].append(self.NCC)
        self.result['SSIM'].append(self.SSIM)
        self.result['MI'].append(self.MI)

    def print_results(self, opt):
        """
        打印评估结果
        
        参数:
            opt: 选项对象
            
        功能:
            格式化显示所有评估指标的平均值
        """
        message = ''
        message += '----------------- Evaluations ---------------\n'
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format('FWHM:', str(np.mean(self.result['FWHM'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('Contrast [db]:', str(np.mean(self.result['CR'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('CNR:', str(np.mean(self.result['CNR'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('sSNR:', str(np.mean(self.result['sSNR'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('GCNR:', str(np.mean(self.result['GCNR'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('L1 loss:', str(np.mean(self.result['L1loss'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('L2 loss:', str(np.mean(self.result['L2loss'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('PSNR:', str(np.mean(self.result['PSNR'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('SSIM:', str(np.mean(self.result['SSIM'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('MI:', str(np.mean(self.result['MI'])), comment)
        message += '{:>25}: {:<30}{}\n'.format('NCC:', str(np.mean(self.result['NCC'])), comment)
        message += '----------------- End -------------------'
        print(message)

    def save_result(self, opt, message):
        """
        保存评估结果到文件
        
        参数:
            opt: 选项对象，包含保存路径信息
            message: 要保存的结果消息
            
        功能:
            将评估结果保存到实验目录的result.txt文件
        """
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        file_name = os.path.join(expr_dir, 'result.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


# ===== 使用说明和总结 =====
"""
utils/metrics.py 在AUGAN项目中的作用：

1. 专业超声评估：
   - 针对超声成像特点设计的专业指标
   - 支持点目标和体模目标的不同评估模式
   - 考虑超声成像的物理特性（散斑、分辨率等）

2. 综合质量评估：
   - 结合传统图像指标（PSNR, SSIM）
   - 信息论指标（互信息）
   - 统计指标（CNR, GCNR）

3. 评估流程：
   - 自动ROI检测和分析
   - 多指标并行计算
   - 结果统计和保存

4. 应用场景：
   - 训练过程中的模型性能监控
   - 测试阶段的最终质量评估
   - 不同模型间的性能对比

使用示例：
```python
evaluator = image_evaluation()
evaluator.evaluate(generated_img, target_img, opt, plane_wave_data, test_type=1, i=0)
evaluator.print_results(opt)
evaluator.save_result(opt, message)
```

关键改进：
1. 量化处理避免sklearn警告
2. 鲁棒的6dB分辨率计算
3. 多种测试类型的自适应ROI设置
4. 完整的结果管理系统
"""