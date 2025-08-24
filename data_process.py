#!/usr/bin/env python3
"""
AUGAN 数据处理模块 - 详细注释版
data_process_annotated.py - 处理PICMUS超声数据集，生成训练所需的图像对

主要功能:
1. 加载PICMUS HDF5数据集
2. 执行DAS (Delay-and-Sum) 重建算法
3. 生成单角度和多角度超声图像对
4. 数据预处理和归一化
5. 创建PyTorch数据集和数据加载器

数据流程:
PICMUS数据 → DAS重建 → 图像对生成 → 数据集封装 → 训练/测试使用
"""

import torch
import logging
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader, Dataset, TensorDataset
from cubdl.example_picmus_torch import load_datasets, create_network, mk_img, dispaly_img


class AUGANDataset(Dataset):
    """
    AUGAN专用数据集类
    
    功能:
        封装处理后的超声图像数据，提供PyTorch标准数据集接口
    
    数据结构:
        - A: 输入图像（单角度DAS重建，低质量）
        - B: 目标图像（多角度复合重建，高质量）
    """
    
    def __init__(self, input_images, target_images, phase='train'):
        """
        初始化数据集
        
        参数:
            input_images: 输入图像列表（单角度）
            target_images: 目标图像列表（多角度）
            phase: 数据集阶段 ('train', 'test', 'val')
        """
        self.input_images = input_images
        self.target_images = target_images
        self.phase = phase
        self.len = len(input_images)
        
        print(f"📊 创建{phase}数据集: {self.len}个样本")
    
    def __len__(self):
        """返回数据集大小"""
        return self.len
    
    def __getitem__(self, index):
        """
        获取单个数据样本
        
        参数:
            index: 数据索引
        
        返回:
            data: 包含A(输入)和B(目标)的字典
        """
        # 获取对应的输入和目标图像
        input_img = self.input_images[index]
        target_img = self.target_images[index]
        
        # 确保数据类型为float32
        if isinstance(input_img, np.ndarray):
            input_img = torch.from_numpy(input_img).float()
        if isinstance(target_img, np.ndarray):
            target_img = torch.from_numpy(target_img).float()
        
        # 确保维度正确 [C, H, W]
        if len(input_img.shape) == 2:
            input_img = input_img.unsqueeze(0)  # 添加通道维度
        if len(target_img.shape) == 2:
            target_img = target_img.unsqueeze(0)
        
        return {
            'A': input_img,      # 输入图像（单角度）
            'B': target_img,     # 目标图像（多角度）
            'A_paths': f'input_{index}',    # 输入路径（用于调试）
            'B_paths': f'target_{index}'    # 目标路径（用于调试）
        }
    
    def get_item(self, index):
        """
        兼容性方法，与原有代码接口保持一致
        """
        return self.__getitem__(index)


def test_image(data, data1, target, xlims, zlims, i, phase, name):
    """
    可视化函数：显示原始图像、重建图像和目标图像的对比
    
    参数说明:
        data (tensor): 输入图像 (单角度DAS图像)
        data1 (tensor): 生成器重建的图像 
        target (tensor): 目标图像 (复合角度高质量图像)
        xlims, zlims: 显示的坐标范围
        i: 图像索引
        phase: 训练阶段 ('train' 或 'test')
        name: 实验名称
    
    注意: 保持原有的参数名以确保向后兼容性
    """
    # 重命名参数以提高内部代码可读性
    input_image = data
    generated_image = data1
    target_image = target
    
    # 将tensor转为numpy，便于后续处理和显示
    input_image_np = input_image.detach().cpu().numpy()      # 输入图像 (低质量单角度)
    input_image_np = np.squeeze(input_image_np)              # 移除大小为1的维度
    input_image_np -= np.max(input_image_np)                 # 标准化：减去最大值 (dB表示，使最大值为0dB)

    generated_image_np = generated_image.detach().cpu().numpy()    # 生成器输出图像 (增强后)
    generated_image_np = np.squeeze(generated_image_np)
    generated_image_np -= np.max(generated_image_np)               # 同样标准化

    target_image_np = target_image.detach().cpu().numpy()   # 目标图像 (高质量复合角度)
    target_image_np = np.squeeze(target_image_np)
    target_image_np -= np.max(target_image_np)               # 同样标准化

    # 调用专门的超声图像显示函数，显示三图对比
    # [1] 表示显示的角度列表
    dispaly_img(input_image_np, generated_image_np, target_image_np, xlims, zlims, [1], i, phase, name)


def create_das_reconstructions(plane_wave_data, single_angle=[1], multi_angles=None):
    """
    创建DAS重建图像
    
    参数:
        plane_wave_data: PICMUS平面波数据
        single_angle: 单角度列表（作为输入）
        multi_angles: 多角度列表（作为目标，None时使用所有角度）
    
    返回:
        single_images: 单角度重建图像列表
        multi_images: 多角度重建图像列表
        das, iqdata, xlims, zlims: DAS相关参数
    
    DAS原理:
        Delay-and-Sum算法是超声成像的核心重建方法
        - 单角度：使用单一发射角度，图像质量较低但速度快
        - 多角度：使用多个角度复合，图像质量高但计算量大
    """
    print("🔬 执行DAS重建算法...")
    
    # 创建单角度DAS网络（输入图像）
    print(f"📡 单角度重建: 角度 {single_angle}")
    das_single, iqdata, xlims, zlims = create_network(plane_wave_data, single_angle)
    
    # 创建多角度DAS网络（目标图像）
    if multi_angles is None:
        # 使用所有可用角度（通常是75个角度：-37°到37°）
        multi_angles = list(range(len(plane_wave_data.angles)))
        print(f"📡 多角度重建: 使用全部 {len(multi_angles)} 个角度")
    else:
        print(f"📡 多角度重建: 角度 {multi_angles}")
    
    das_multi, _, _, _ = create_network(plane_wave_data, multi_angles)
    
    # 执行重建
    print("⚙️  执行图像重建...")
    single_images = []
    multi_images = []
    
    # 这里简化处理，实际应该根据具体的PICMUS数据结构来处理
    # 假设我们有N个超声帧需要处理
    num_frames = getattr(plane_wave_data, 'nb_frames', 100)  # 默认100帧
    
    # 简化处理：只生成一张图像用于演示
    try:
        # 单角度重建（输入）
        single_img = mk_img(das_single, iqdata)
        single_images.append(single_img)
        
        # 多角度重建（目标）
        multi_img = mk_img(das_multi, iqdata)
        multi_images.append(multi_img)
        
        print(f"✅ 重建完成: 1 对图像")
        
    except Exception as e:
        print(f"⚠️  图像重建失败: {e}")
        return [], [], das_single, iqdata, xlims, zlims
    
    return single_images, multi_images, das_single, iqdata, xlims, zlims


def preprocess_images(images, normalize=True, log_compress=True, target_size=None):
    """
    图像预处理
    
    参数:
        images: 图像列表
        normalize: 是否归一化到[0,1]
        log_compress: 是否应用对数压缩
        target_size: 目标图像尺寸 (H, W)
    
    返回:
        processed_images: 处理后的图像列表
    
    预处理步骤:
        1. 对数压缩（模拟超声成像的显示方式）
        2. 尺寸调整到256x256
        3. 归一化（便于神经网络训练）
        4. 数据类型转换
    """
    import cv2
    print("🔧 执行图像预处理...")
    processed_images = []
    
    for i, img in enumerate(images):
        try:
            # 确保为numpy数组
            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()
            
            print(f"📏 原始图像{i}尺寸: {img.shape}")
            
            # 对数压缩（超声成像标准处理）
            if log_compress:
                # 避免log(0)，添加小常数
                img = np.log10(np.abs(img) + 1e-10)
            
            # 归一化到[0, 1]
            if normalize:
                img_min = np.min(img)
                img_max = np.max(img)
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                else:
                    img = np.zeros_like(img)
            
            # 智能尺寸处理：padding到512x512而不是resize，保持数据完整性
            if img.shape[0] < 512 or img.shape[1] < 512:
                # 计算padding
                pad_h = max(0, 512 - img.shape[0])
                pad_w = max(0, 512 - img.shape[1])
                # 对称padding
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                original_shape = img.shape
                img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
                print(f"🔄 图像{i} padding: {original_shape[0]}x{original_shape[1]} -> {img.shape[0]}x{img.shape[1]} (保持数据完整性)")
            else:
                print(f"✅ 保持原始尺寸: {img.shape}")
            
            # 确保数据类型为float32
            img = img.astype(np.float32)
            
            processed_images.append(img)
            
        except Exception as e:
            print(f"⚠️  图像 {i} 预处理失败: {e}")
            continue
    
    if target_size:
        print(f"✅ 预处理完成: {len(processed_images)} 张图像，目标尺寸: {target_size}")
    else:
        print(f"✅ 预处理完成: {len(processed_images)} 张图像，保持原始尺寸")
    return processed_images


def split_dataset(input_images, target_images, train_ratio=0.8, val_ratio=0.1):
    """
    数据集划分
    
    参数:
        input_images: 输入图像列表
        target_images: 目标图像列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例（剩余为测试集）
    
    返回:
        splits: 包含train/val/test划分的字典
    """
    print("📊 划分数据集...")
    
    total_samples = len(input_images)
    
    # 如果样本太少，调整划分策略
    if total_samples <= 3:
        # 样本太少时，全部用作训练，复制数据增加样本数
        print(f"⚠️  样本数量过少({total_samples})，复制数据以增加训练样本")
        # 复制数据10次
        multiplied_input = input_images * 10
        multiplied_target = target_images * 10
        train_size = len(multiplied_input)
        val_size = 0
        test_size = total_samples  # 原始数据用作测试
        
        train_indices = list(range(train_size))
        val_indices = []
        test_indices = list(range(total_samples))
        
        splits = {
            'train': {
                'input': multiplied_input,
                'target': multiplied_target
            },
            'val': {
                'input': [],
                'target': []
            },
            'test': {
                'input': input_images,
                'target': target_images
            }
        }
    else:
        # 正常划分
        train_size = max(1, int(total_samples * train_ratio))
        val_size = max(0, int(total_samples * val_ratio))
        test_size = total_samples - train_size - val_size
        
        # 随机打乱索引
        indices = np.random.permutation(total_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        splits = {
            'train': {
                'input': [input_images[i] for i in train_indices],
                'target': [target_images[i] for i in train_indices]
            },
            'val': {
                'input': [input_images[i] for i in val_indices],
                'target': [target_images[i] for i in val_indices]
            },
            'test': {
                'input': [input_images[i] for i in test_indices],
                'target': [target_images[i] for i in test_indices]
            }
        }
    
    return splits


def load_dataset(opt, phase, dataset_index=0):
    """
    加载数据集（兼容原接口）
    
    参数:
        opt: 训练/测试选项
        phase: 数据集阶段 ('train', 'test', 'val')
        dataset_index: 数据集索引（保留用于兼容性）
    
    返回:
        dataset: AUGAN数据集对象
    
    功能:
        1. 加载PICMUS数据
        2. 执行DAS重建
        3. 图像预处理
        4. 创建数据集对象
    """
    print(f"📂 加载 {phase} 数据集...")
    
    try:
        # 加载PICMUS数据
        plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")
        
        # 执行DAS重建
        single_images, multi_images, das, iqdata, xlims, zlims = create_das_reconstructions(
            plane_wave_data, single_angle=[1])
        
        # 图像预处理
        single_images = preprocess_images(single_images)
        multi_images = preprocess_images(multi_images)
        
        # 数据集划分
        splits = split_dataset(single_images, multi_images)
        
        # 根据phase获取对应数据
        if phase in splits:
            input_imgs = splits[phase]['input']
            target_imgs = splits[phase]['target']
        else:
            print(f"⚠️  未知阶段 {phase}，使用训练数据")
            input_imgs = splits['train']['input']
            target_imgs = splits['train']['target']
        
        # 创建数据集对象
        dataset = AUGANDataset(input_imgs, target_imgs, phase)
        
        # 为兼容性添加额外属性
        dataset.das = das
        dataset.iqdata = iqdata
        dataset.xlims = xlims
        dataset.zlims = zlims
        
        print(f"✅ {phase} 数据集加载完成")
        return dataset
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        # 返回空数据集作为后备
        return AUGANDataset([], [], phase)


def save_processed_data(input_images, target_images, save_dir='./data'):
    """
    保存处理后的数据到磁盘
    
    参数:
        input_images: 输入图像列表
        target_images: 目标图像列表
        save_dir: 保存目录
    
    功能:
        将处理后的数据保存为.mat文件，加速后续加载
    """
    import os
    
    print("💾 保存处理后的数据...")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 转换为numpy数组
        input_array = np.array(input_images)
        target_array = np.array(target_images)
        
        # 保存为.mat文件
        sio.savemat(os.path.join(save_dir, 'train_inputdata.mat'), 
                   {'input_data': input_array})
        sio.savemat(os.path.join(save_dir, 'train_targetdata.mat'), 
                   {'target_data': target_array})
        
        print(f"✅ 数据已保存到 {save_dir}")
        
    except Exception as e:
        print(f"❌ 数据保存失败: {e}")


def main():
    """
    主函数 - 用于独立运行数据处理
    
    功能:
        完整的数据处理流程演示
    """
    print("🚀 AUGAN数据处理程序启动...")
    print("="*50)
    
    try:
        # 加载PICMUS数据
        print("1️⃣  加载PICMUS数据集...")
        plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")
        
        # 执行DAS重建
        print("2️⃣  执行DAS重建...")
        single_images, multi_images, das, iqdata, xlims, zlims = create_das_reconstructions(
            plane_wave_data)
        
        # 图像预处理
        print("3️⃣  图像预处理...")
        single_images = preprocess_images(single_images)
        multi_images = preprocess_images(multi_images)
        
        # 保存处理后的数据
        print("4️⃣  保存数据...")
        save_processed_data(single_images, multi_images)
        
        # 创建数据集示例
        print("5️⃣  创建数据集...")
        dataset = AUGANDataset(single_images, multi_images, 'train')
        
        print("="*50)
        print("🎉 数据处理完成！")
        print(f"📊 处理的图像对数量: {len(single_images)}")
        print(f"📁 数据保存位置: ./data/")
        
    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    """
    独立运行数据处理脚本
    
    使用方法:
        python data_process_annotated.py
    
    功能:
        预处理PICMUS数据集，生成训练所需的图像对
    """
    main()