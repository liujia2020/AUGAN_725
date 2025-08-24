#!/usr/bin/env python3
"""
AUGAN SSIM损失函数模块 - 详细注释版
pytorch_ssim/__init__.py - 实现结构相似性指数(SSIM)的计算和损失函数
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

""" 
SSIM (Structural Similarity Index) 结构相似性指数
用于评估两幅图像的相似度，比MSE更符合人眼感知

SSIM优势:
1. 考虑图像的亮度、对比度和结构信息
2. 与人眼感知更相关
3. 取值范围[-1, 1]，越接近1越相似

在AUGAN中的作用:
- 作为损失函数的一部分，提升生成图像质量
- 评估生成图像与目标图像的相似度
"""

def gaussian(window_size, sigma):
    """
    生成高斯核
    
    参数:
        window_size (int): 窗口大小
        sigma (float): 高斯分布标准差
    
    返回:
        归一化的高斯核
    
    作用:
        SSIM计算需要高斯加权，模拟人眼对不同位置的敏感度
    """
    gauss = torch.Tensor([
        exp(-(x - window_size//2)**2/float(2*sigma**2)) 
        for x in range(window_size)
    ])
    return gauss/gauss.sum()  # 归一化

def create_window(window_size, channel):
    """
    创建2D高斯窗口
    
    参数:
        window_size (int): 窗口大小（通常为11）
        channel (int): 图像通道数
    
    返回:
        2D高斯窗口tensor
    
    实现:
        1. 生成1D高斯核
        2. 外积得到2D高斯窗口
        3. 扩展到指定通道数
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # 转为列向量
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # 外积
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    计算SSIM的核心函数
    
    参数:
        img1, img2: 输入图像
        window: 高斯窗口
        window_size: 窗口大小
        channel: 通道数
        size_average: 是否对结果取平均
    
    返回:
        SSIM值
    
    SSIM公式:
        SSIM(x,y) = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
        其中：
        - μx, μy: 图像均值
        - σx², σy²: 图像方差  
        - σxy: 协方差
        - C1, C2: 稳定常数
    """
    # 计算局部均值（使用高斯加权）
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    # 计算均值的平方和乘积
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    # SSIM稳定常数
    C1 = 0.01**2  # (K1*L)² where K1=0.01, L=1 for normalized images
    C2 = 0.03**2  # (K2*L)² where K2=0.03, L=1

    # 计算SSIM
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    """
    SSIM损失函数类
    
    使用方式:
        ssim_loss = SSIM()
        loss = 1 - ssim_loss(img1, img2)  # 转换为损失（越小越好）
    
    特点:
        - 可以作为PyTorch模块使用
        - 支持自动梯度计算
        - 可以与其他损失函数结合使用
    """
    
    def __init__(self, window_size=11, size_average=True):
        """
        初始化SSIM模块
        
        参数:
            window_size (int): 高斯窗口大小，默认11
            size_average (bool): 是否对结果取平均
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        前向传播计算SSIM
        
        参数:
            img1, img2: 输入图像 [B, C, H, W]
        
        返回:
            SSIM值，范围[-1, 1]
        """
        (_, channel, _, _) = img1.size()

        # 检查窗口是否需要重新创建
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            # 如果图像在GPU上，窗口也需要在GPU上
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size=11, size_average=True):
    """
    SSIM计算的函数式接口
    
    参数:
        img1, img2: 输入图像
        window_size: 窗口大小
        size_average: 是否取平均
    
    返回:
        SSIM值
    
    使用:
        similarity = ssim(pred_img, target_img)
        ssim_loss = 1 - similarity  # 转换为损失
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)