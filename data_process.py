#!/usr/bin/env python3
"""
AUGAN 数据处理模块 - MVP精简版
只保留训练必需的核心功能
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from cubdl.example_picmus_torch import load_datasets, create_network, mk_img, dispaly_img


class AUGANDataset(Dataset):
    """AUGAN数据集类 - 简化版"""
    
    def __init__(self, input_images, target_images, phase='train'):
        self.input_images = input_images
        self.target_images = target_images
        self.phase = phase
        self.len = len(input_images)
        print(f"创建{phase}数据集: {self.len}个样本")
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        input_img = self.input_images[index]
        target_img = self.target_images[index]
        
        # 确保是float32张量
        if isinstance(input_img, np.ndarray):
            input_img = torch.from_numpy(input_img).float()
        if isinstance(target_img, np.ndarray):
            target_img = torch.from_numpy(target_img).float()
        
        # 添加通道维度
        if len(input_img.shape) == 2:
            input_img = input_img.unsqueeze(0)
        if len(target_img.shape) == 2:
            target_img = target_img.unsqueeze(0)
        
        return {
            'A': input_img,
            'B': target_img,
            'A_paths': f'input_{index}',
            'B_paths': f'target_{index}'
        }
    
    def get_item(self, index):
        """兼容原接口"""
        return self.__getitem__(index)


def test_image(data, data1, target, xlims, zlims, i, phase, name):
    """可视化函数 - 保留用于训练监控"""
    # 转换为numpy
    input_image_np = data.detach().cpu().numpy()
    input_image_np = np.squeeze(input_image_np)
    input_image_np -= np.max(input_image_np)

    generated_image_np = data1.detach().cpu().numpy()
    generated_image_np = np.squeeze(generated_image_np)
    generated_image_np -= np.max(generated_image_np)

    target_image_np = target.detach().cpu().numpy()
    target_image_np = np.squeeze(target_image_np)
    target_image_np -= np.max(target_image_np)

    # 调用PICMUS显示函数
    dispaly_img(input_image_np, generated_image_np, target_image_np, 
                xlims, zlims, [1], i, phase, name)


def load_dataset(opt, phase, dataset_index=0):
    """
    加载数据集 - 极简版
    生成75对训练图像（每个角度一对）
    """
    print(f"加载 {phase} 数据集...")
    
    # 加载PICMUS数据
    plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")
    
    # 获取多角度DAS（目标图像）
    das_multi, iqdata, xlims, zlims = create_network(plane_wave_data, list(range(75)))
    multi_img = mk_img(das_multi, iqdata)
    
    # 简单预处理
    def preprocess(img):
        # 归一化到[0,1]
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        return img.astype(np.float32)
    
    # 生成多个训练对（使用不同的单角度）
    input_imgs = []
    target_imgs = []
    
    # 为每个角度创建一对图像
    for angle_idx in range(0, 75, 5):  # 每5个角度取一个，生成15对
        # 单角度重建
        das_single, _, _, _ = create_network(plane_wave_data, [angle_idx])
        single_img = mk_img(das_single, iqdata)
        
        # 预处理
        single_img = preprocess(single_img)
        multi_img_processed = preprocess(multi_img)
        
        input_imgs.append(single_img)
        target_imgs.append(multi_img_processed)
    
    print(f"生成了 {len(input_imgs)} 对图像")
    
    # 创建数据集
    dataset = AUGANDataset(input_imgs, target_imgs, phase)
    
    # 添加属性用于兼容
    dataset.das = das_multi
    dataset.iqdata = iqdata
    dataset.xlims = xlims
    dataset.zlims = zlims
    
    return dataset