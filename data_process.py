# import torch
# import logging
# import numpy as np
# import scipy.io as sio
# from torch.utils.data import DataLoader,Dataset,TensorDataset
# from cubdl.example_picmus_torch import load_datasets,create_network,mk_img,dispaly_img

# """  Display the original image, reconstructed image and target image"""
# def test_image(data, data1, target, xlims, zlims,i,phase,name):
#     data = data.detach().numpy()
#     data = np.squeeze(data)
#     data -= np.max(data)

#     data1 = data1.detach().numpy()
#     data1 = np.squeeze(data1)
#     data1 -= np.max(data1)

#     target = target.detach().numpy()
#     target = np.squeeze(target)
#     target -= np.max(target)

#     dispaly_img(data, data1, target, xlims, zlims, [1], i, phase, name)


# """Data augment to image dataset"""
# class Img_Dataset(Dataset):
#     def __init__(self,input_tensor,target_tensor,num,height,width,test_type):
#         super(Img_Dataset,self).__init__()
#         self.a = torch.Tensor(input_tensor)

#         """Adding Gussain noise to the image"""
#         def Gussain_nosie_add(input_tensor):
#             noise = np.random.normal(0,1,(height,width))
#             noise = torch.Tensor(noise)
#             normal_tensor = torch.add(input_tensor,noise)
#             return normal_tensor
#         self.normal_tensor = (Gussain_nosie_add(self.a))
#         self.input_tensor = torch.Tensor(input_tensor)
#         self.target_data = torch.Tensor(target_tensor)

#         """Flip the image"""
#         self.updownflip_img = np.zeros((num,height,width),dtype='float32')
#         self.leftrightflip_img = np.zeros((num,height,width),dtype='float32')

#         for index1 in range(num):
#             a = np.flip(input_tensor[index1,:,:],0).copy()
#             self.updownflip_img[index1,:,:] = torch.Tensor(a)
#             b = np.flip(input_tensor[index1, :, :], 1).copy()
#             self.leftrightflip_img[index1,:,:] = torch.Tensor(b)

#         self.updownflip_img = (torch.from_numpy(self.updownflip_img))
#         self.leftrightflip_img = (torch.from_numpy(self.leftrightflip_img))

#         self.input_data = torch.cat([self.input_tensor,self.normal_tensor,self.leftrightflip_img,self.updownflip_img])
#         # self.input_data = torch.cat([self.input_tensor, self.normal_tensor])
#         self.len = self.input_data.shape[0]
#         # if test_type == 0:
#         #     sio.savemat('data/traindata.mat', {'input_data':self.input_data,'target_data':self.target_data5})
#         # else:
#         #     sio.savemat('data/testdata.mat', {'data': self.input_data,'target_data':self.target_data5})

#         if test_type == 0: # 0 represents the training mode
#             input_data = self.input_tensor.numpy()
#             target_data = self.target_data.numpy()
#             sio.savemat('data/train_inputdata.mat', {'input_data': input_data})
#             sio.savemat('data/train_targetdata.mat', {'target_data': target_data})
#         else: # 1-5 represent the test mode
#             input_data = self.input_tensor.numpy()
#             target_data = self.target_data.numpy()
#             sio.savemat('data/test_inputdata.mat', {'data': self.input_data})
#             sio.savemat('data/test_targetdata.mat', {'target_data': self.target_data})

#     # def __getitem__(self,index):
#     #     return self.input_data[index], self.target_data[index]

#     # def get_item(self, index):
#     # # """兼容原接口"""
#     #     return self.__getitem__(index)
    
#     def __getitem__(self, index):
#         # 原来的代码：
#         # return self.input_data[index], self.target_data[index]
        
#         # 🎯 替换为：
#         input_img = self.input_data[index]
#         target_img = self.target_data[index]
        
#         # 确保维度正确
#         if len(input_img.shape) == 2:
#             input_img = input_img.unsqueeze(0)
#         if len(target_img.shape) == 2:
#             target_img = target_img.unsqueeze(0)
            
#         return {
#             'A': input_img,
#             'B': target_img,
#             'A_paths': f'input_{index}',
#             'B_paths': f'target_{index}'
#         }
    
#     def __len__(self):
#         return self.len

# def load_dataset(opt, phase, test_type):
#     # load mat data
#     s1= []
#     mat_num = 6
#     if test_type == 0: # train mode
#         s1 = [ "simulation_resolution_distorsion_iq","simulation_contrast_speckle_iq" ,"experiments_resolution_distorsion_iq","experiments_contrast_speckle_iq",
#             "experiments_carotid_long_iq","experiments_carotid_cross_iq"]
#         mat_num = 6
#     # different test type
#     elif test_type == 1:
#         s1 = ["simulation_resolution_distorsion_iq"]
#         mat_num = 1
#     elif test_type == 2:
#         s1 = ["simulation_contrast_speckle_iq"]
#         mat_num = 1
#     elif test_type == 3:
#         s1 = ["experiments_resolution_distorsion_iq"]
#         mat_num = 1
#     elif test_type == 4:
#         s1 = ["experiments_contrast_speckle_iq"]
#         mat_num = 1
#     elif test_type == 5:
#         s1 = ["experiments_carotid_long_iq","experiments_carotid_cross_iq"]
#         mat_num = 2
#     elif test_type == 6:
#         s1 = ["simulation_resolution_distorsion_iq","simulation_contrast_speckle_iq" ,"experiments_resolution_distorsion_iq","experiments_contrast_speckle_iq",
#             "experiments_carotid_long_iq","experiments_carotid_cross_iq"]
#         mat_num = 6
#     index = 1
#     i = 0
#     if opt.load:
#         for s in s1:

#             mat_str = ''
#             if phase == 'train':
#                 mat_str = './img_data1/' + s + '_train'
#             elif phase == 'test':
#                 mat_str = './img_data1/' + s + '_test'
#             mat_str1 = s + '_data'
#             try:
#                 a = sio.loadmat(mat_str)
#             except FileNotFoundError:
#                 logging.info(f'file not exist!')
#                 # i = i + 1
#                 mat_num = mat_num - 1
#                 continue
#             if index == 1:
#                 n,m,j = a[mat_str1].shape
#                 # We adjusted the size of the image from 508*387 to 512*384
#                 img_single_data1 = np.zeros((n-1,m+4,j))
#                 img_compound_data1 = np.zeros((1,m+4,j))
#                 img_single_data2 = np.zeros((6,n-1,m+4,j-3))
#                 img_compound_data2 = np.zeros((6,1,m+4,j-3))
#                 index = 0
#             img_single_data1[:, 0:m, :] = a[mat_str1][0:(n-1), 0:m, :] # 前60张
#             img_compound_data1[:, 0:m, :] = a[mat_str1][(n-1):n, 0:m, :]
#             img_single_data1[:,m:(m+4),:] = a[mat_str1][0:(n-1),(m-4):m,:]
#             img_compound_data1[:, m:(m+4), :] = a[mat_str1][(n-1):n, (m-4):m, :]

#             img_single_data2[i,:,:,:] = img_single_data1[:,:,0:j-3]
#             img_compound_data2[i,:,:,:] = img_compound_data1[:,:,0:j-3]
#             i = i + 1
#             logging.info(f'{s} data loaded')
#     logging.info(f'data loaded completed')

#     img_single_data = img_single_data2[0:mat_num,:,:]
#     img_compound_data = img_compound_data2[0:mat_num,:,:]
#     # img_single_data = img_single_data2[:, :, :, :]
#     # img_compound_data = img_compound_data2[:, :, :, :]
#     n,m ,i ,j = img_single_data.shape
#     img_single_data = np.reshape(img_single_data,(-1,i,j))
#     num = img_single_data.shape[0]

#     img_com75_data = img_compound_data[:, 0, :, :]
#     # data augment to target images
#     img_com75_data1 = np.empty((n * m, i, j))
#     img_com75_data2 = np.empty((n * m, i, j))
#     img_com75_data3 = np.empty((n * m, i, j))
#     img_com75_data4 = np.empty((n * m, i, j))



#     for i1 in range(n):
#         i2 = i1*m
#         i3 = (i1+1 )*m
#         com75_data = np.expand_dims(img_com75_data[i1, :, :], axis=0)
#         a75 = np.repeat(com75_data, m, axis=0)
#         img_com75_data1[i2:i3, :, :] = a75

#         for index1 in range(num):
#             img_com75_data2[index1, :, :] = np.flip(img_com75_data1[index1, :, :], 0)
#             img_com75_data3[index1, :, :] = np.flip(img_com75_data1[index1, :, :], 1)

#     img_com75_data1 = np.concatenate((img_com75_data1,img_com75_data1,img_com75_data3,img_com75_data2), axis=0)

#     img_dataset = Img_Dataset(input_tensor=img_single_data, target_tensor= img_com75_data1 , num=num, height=i, width=j, test_type=test_type)
#     return img_dataset

#!/usr/bin/env python3
"""
修复后的data_process.py
解决get_item方法不存在的问题
"""

import torch
import logging
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader,Dataset,TensorDataset
from cubdl.example_picmus_torch import load_datasets,create_network,mk_img,dispaly_img

"""  Display the original image, reconstructed image and target image"""
def test_image(data, data1, target, xlims, zlims,i,phase,name):
    data = data.detach().numpy()
    data = np.squeeze(data)
    data -= np.max(data)

    data1 = data1.detach().numpy()
    data1 = np.squeeze(data1)
    data1 -= np.max(data1)

    target = target.detach().numpy()
    target = np.squeeze(target)
    target -= np.max(target)

    dispaly_img(data, data1, target, xlims, zlims, [1], i, phase, name)


"""Data augment to image dataset"""
class Img_Dataset(Dataset):
    def __init__(self,input_tensor,target_tensor,num,height,width,test_type):
        super(Img_Dataset,self).__init__()
        self.a = torch.Tensor(input_tensor)

        """Adding Gussain noise to the image"""
        def Gussain_nosie_add(input_tensor):
            noise = np.random.normal(0,1,(height,width))
            noise = torch.Tensor(noise)
            normal_tensor = torch.add(input_tensor,noise)
            return normal_tensor
        self.normal_tensor = (Gussain_nosie_add(self.a))
        self.input_tensor = torch.Tensor(input_tensor)
        self.target_data = torch.Tensor(target_tensor)

        """Flip the image"""
        self.updownflip_img = np.zeros((num,height,width),dtype='float32')
        self.leftrightflip_img = np.zeros((num,height,width),dtype='float32')

        for index1 in range(num):
            a = np.flip(input_tensor[index1,:,:],0).copy()
            self.updownflip_img[index1,:,:] = torch.Tensor(a)
            b = np.flip(input_tensor[index1, :, :], 1).copy()
            self.leftrightflip_img[index1,:,:] = torch.Tensor(b)

        self.updownflip_img = (torch.from_numpy(self.updownflip_img))
        self.leftrightflip_img = (torch.from_numpy(self.leftrightflip_img))

        self.input_data = torch.cat([self.input_tensor,self.normal_tensor,self.leftrightflip_img,self.updownflip_img])
        # self.input_data = torch.cat([self.input_tensor, self.normal_tensor])
        self.len = self.input_data.shape[0]
        
        # 保存数据到MAT文件
        if test_type == 0: # 0 represents the training mode
            input_data = self.input_tensor.numpy()
            target_data = self.target_data.numpy()
            sio.savemat('data/train_inputdata.mat', {'input_data': input_data})
            sio.savemat('data/train_targetdata.mat', {'target_data': target_data})
        else: # 1-5 represent the test mode
            input_data = self.input_tensor.numpy()
            target_data = self.target_data.numpy()
            sio.savemat('data/test_inputdata.mat', {'data': self.input_data})
            sio.savemat('data/test_targetdata.mat', {'target_data': self.target_data})

    def __getitem__(self, index):
        """
        获取数据项的标准方法
        返回符合Pix2Pix模型期望的数据格式
        """
        # 🎯 修复关键问题：确保索引不越界
        if index >= len(self.input_data):
            index = index % len(self.input_data)
            
        input_img = self.input_data[index]
        
        # 🎯 修复关键问题：目标数据要根据增强后的索引来计算
        # 对于增强数据，都使用原始目标图像
        original_index = index % len(self.target_data)
        target_img = self.target_data[original_index]
        
        # 确保维度正确
        if len(input_img.shape) == 2:
            input_img = input_img.unsqueeze(0)  # 添加通道维度
        if len(target_img.shape) == 2:
            target_img = target_img.unsqueeze(0)  # 添加通道维度
            
        return {
            'A': input_img,
            'B': target_img,
            'A_paths': f'input_{index}',
            'B_paths': f'target_{original_index}'
        }

    def get_item(self, index):
        """
        🎯 添加get_item方法，解决测试脚本中的调用问题
        这是为了兼容原有代码中使用get_item的地方
        """
        return self.__getitem__(index)
    
    def __len__(self):
        return self.len


def load_dataset(opt, phase, test_type):
    """
    加载数据集的主函数
    """
    # load mat data
    s1= []
    mat_num = 6
    if test_type == 0: # train mode
        s1 = [ "simulation_resolution_distorsion_iq","simulation_contrast_speckle_iq" ,"experiments_resolution_distorsion_iq","experiments_contrast_speckle_iq",
            "experiments_carotid_long_iq","experiments_carotid_cross_iq"]
        mat_num = 6
    # different test type
    elif test_type == 1:
        s1 = ["simulation_resolution_distorsion_iq"]
        mat_num = 1
    elif test_type == 2:
        s1 = ["simulation_contrast_speckle_iq"]
        mat_num = 1
    elif test_type == 3:
        s1 = ["experiments_resolution_distorsion_iq"]
        mat_num = 1
    elif test_type == 4:
        s1 = ["experiments_contrast_speckle_iq"]
        mat_num = 1
    elif test_type == 5:
        s1 = ["experiments_carotid_long_iq","experiments_carotid_cross_iq"]
        mat_num = 2
    elif test_type == 6:
        s1 = ["simulation_resolution_distorsion_iq","simulation_contrast_speckle_iq" ,"experiments_resolution_distorsion_iq","experiments_contrast_speckle_iq",
            "experiments_carotid_long_iq","experiments_carotid_cross_iq"]
        mat_num = 6
    
    index = 1
    i = 0
    
    if opt.load:
        for s in s1:
            mat_str = ''
            if phase == 'train':
                mat_str = './img_data1/' + s + '_train'
            elif phase == 'test':
                mat_str = './img_data1/' + s + '_test'
            mat_str1 = s + '_data'
            
            try:
                a = sio.loadmat(mat_str)
                logging.info(f'{s} data loaded')
            except FileNotFoundError:
                logging.info(f'file not exist!')
                mat_num = mat_num - 1
                continue
                
            if index == 1:
                n,m,j = a[mat_str1].shape
                # We adjusted the size of the image from 508*387 to 512*384
                img_single_data1 = np.zeros((n-1,m+4,j))
                img_compound_data1 = np.zeros((1,m+4,j))
                img_single_data2 = np.zeros((6,n-1,m+4,j-3))
                img_compound_data2 = np.zeros((6,1,m+4,j-3))
                index = 0
                
            img_single_data1[:, 0:m, :] = a[mat_str1][0:(n-1), 0:m, :] # 前60张
            img_compound_data1[:, 0:m, :] = a[mat_str1][(n-1):n, 0:m, :]
            img_single_data1[:,m:(m+4),:] = a[mat_str1][0:(n-1),(m-4):m,:]
            img_compound_data1[:, m:(m+4), :] = a[mat_str1][(n-1):n, (m-4):m, :]

            img_single_data2[i,:,:,:] = img_single_data1[:,:,0:j-3]
            img_compound_data2[i,:,:,:] = img_compound_data1[:,:,0:j-3]
            i = i + 1

    logging.info(f'data loaded completed')

    img_single_data = img_single_data2[0:mat_num,:,:]
    img_compound_data = img_compound_data2[0:mat_num,:,:]
    
    n,m ,i ,j = img_single_data.shape
    img_single_data = np.reshape(img_single_data,(-1,i,j))
    num = img_single_data.shape[0]

    img_com75_data = img_compound_data[:, 0, :, :]
    # data augment to target images
    img_com75_data1 = np.empty((n * m, i, j))
    img_com75_data2 = np.empty((n * m, i, j))
    img_com75_data3 = np.empty((n * m, i, j))
    img_com75_data4 = np.empty((n * m, i, j))

    for i1 in range(n):
        i2 = i1*m
        i3 = (i1+1 )*m
        com75_data = np.expand_dims(img_com75_data[i1, :, :], axis=0)
        a75 = np.repeat(com75_data, m, axis=0)
        img_com75_data1[i2:i3, :, :] = a75

        for index1 in range(num):
            img_com75_data2[index1, :, :] = np.flip(img_com75_data1[index1, :, :], 0)
            img_com75_data3[index1, :, :] = np.flip(img_com75_data1[index1, :, :], 1)

    img_com75_data1 = np.concatenate((img_com75_data1,img_com75_data1,img_com75_data3,img_com75_data2), axis=0)

    img_dataset = Img_Dataset(input_tensor=img_single_data, target_tensor= img_com75_data1 , num=num, height=i, width=j, test_type=test_type)
    return img_dataset