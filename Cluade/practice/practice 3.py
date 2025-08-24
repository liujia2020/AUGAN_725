import torch
import logging
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataloader, Dataset, TensorDataset
from cubdl.example_picmus_torch import load_datasets, create_network, mk_img, dispaly_img

class AUGANDateset(Dataset):
    def __init__(self, input_images, target_images, phase='train'):
        self.input_images = input_images
        self.target_images = target_images
        self.phase = phase
        
        self.len = len(input_images)
        
        print(f"创建{phase}数据集：{self.len}个样本")
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # 获取对应的输入和目标图像
        input_img = self.input_images[index]
        target_img = self.target_images[index]
        
        # 确保数据类型为float32
        if isinstance(input_img, np.ndarray):
            input_img = torch.from_numpy(input_img).float()
        if isinstance(target_img, np.ndarray):
            target_img = torch.from_numpy(target_img).float()
            
        # 确保维度正确
        if len(input_img.shape) == 2:
            input_img = input_img.unsqueeze(0)
        if len(target_img.shape) ==2:
            target_img.shape = target_img.unsqueeze(0)
            
        return {
            'A': input_img,
            'B': target_img,
            'A_paths': f'input_{index}',
            'B_paths': f'target_{index}'
        }
        
    def get_item(self, index):
        return self.__getitem__(index)
    
    
def test_image(data, data1, target, xlims, zlims, i, phase, name):
    input_image = data
    generated_image = data1
    target_image = target
    
    # 讲tensor转为numpy，便于后续处理和显示
    input_image_np = input_image.detach().cpu().numpy()
    input_image_np = np.squeeze(input_image_np)
    input_image_np -= np.max(input_image_np)
    
    generated_image_np = generated_image.detach().cpu().numpy()    # 生成器输出图像 (增强后)
    generated_image_np = np.squeeze(generated_image_np)
    generated_image_np -= np.max(generated_image_np)               # 同样标准化

    target_image_np = target_image.detach().cpu().numpy()   # 目标图像 (高质量复合角度)
    target_image_np = np.squeeze(target_image_np)
    target_image_np -= np.max(target_image_np)               # 同样标准化
    
    dispaly_img(input_image_np, generated_image_np, target_image_np, xlims, zlims, [1], i, phase, name)
    
    
    
def create_das_reconstructions(plane_wave_data, single_angle=[1], multi_angle=None):
    