#!/usr/bin/env python3
"""
AUGAN 工具函数模块 - 详细注释版
util.py - 包含简单的辅助函数，用于目录创建、网络诊断等
"""

from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os

def mkdirs(paths):
    """
    创建空目录（如果不存在）
    
    参数:
        paths (str list) -- 目录路径列表
    
    功能:
        批量创建多个目录，常用于创建实验相关的文件夹
        例如：checkpoints、results、images等目录
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """
    创建单个空目录（如果不存在）
    
    参数:
        path (str) -- 单个目录路径
    
    功能:
        安全地创建目录，避免已存在时报错
        使用os.makedirs()可以创建多层嵌套目录
    """
    if not os.path.exists(path):
        os.makedirs(path)


def diagnose_network(net, name='network'):
    """
    计算并打印网络梯度的平均绝对值
    
    参数:
        net (torch network) -- PyTorch网络模型
        name (str) -- 网络的名称（用于打印标识）
    
    功能:
        调试工具，用于检查网络训练状态：
        - 梯度过大：可能导致梯度爆炸
        - 梯度过小：可能导致梯度消失
        - 梯度为0：参数未更新，网络可能冻结
    
    使用场景:
        在训练过程中定期调用，监控网络健康状态
    """
    mean = 0.0
    count = 0
    
    # 遍历网络的所有参数
    for param in net.parameters():
        if param.grad is not None:  # 确保参数有梯度
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    
    # 计算平均梯度
    if count > 0:
        mean = mean / count
    
    # 打印诊断信息
    print(f"=== 网络诊断: {name} ===")
    print(f"平均梯度绝对值: {mean}")
    
    # 添加梯度状态判断
    if mean > 1.0:
        print("⚠️  梯度较大，可能存在梯度爆炸")
    elif mean < 1e-6:
        print("⚠️  梯度很小，可能存在梯度消失")
    else:
        print("✅ 梯度正常")