#!/usr/bin/env python3
"""
base_options详细注释版本
原文件：options/base_options.py
作用：定义AUGAN训练和测试的基础配置选项，是整个配置管理系统的核心
"""

import argparse
import os
import torch
import models
from utils.util import mkdirs


class BaseOptions():
    """
    基础选项类 - 定义训练和测试时共用的配置选项
    
    这个类是AUGAN项目配置管理的核心，负责：
    1. 解析命令行参数
    2. 管理模型配置选项
    3. 处理GPU设备设置
    4. 保存和显示配置信息
    
    继承关系：
    BaseOptions (基础配置)
    ├── TrainOptions (训练专用配置)
    └── TestOptions (测试专用配置)
    """

    def __init__(self):
        """初始化基础选项类"""
        self.initialized = False  # 标记是否已初始化，防止重复初始化

    def initialize(self, parser):
        """
        定义训练和测试时共用的基础选项
        
        这个方法定义了所有核心配置参数，包括：
        - 实验管理参数
        - 模型架构参数  
        - 网络结构参数
        - 设备和系统参数
        
        参数：
        parser: ArgumentParser对象，用于解析命令行参数
        
        返回：
        parser: 配置好的ArgumentParser对象
        """
        
        # ===== 实验管理参数 =====
        parser.add_argument('--name', type=str, default='experiment_name', 
                           help='实验名称，决定模型和结果的保存位置')
        
        parser.add_argument('--gpu_ids', type=str, default='0', 
                           help='GPU设备ID，例如: 0 或 0,1,2 或 0,2。使用-1表示CPU')
        
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', 
                           help='模型检查点保存目录')
        
        # ===== 模型架构参数 =====
        parser.add_argument('--model', type=str, default='pix2pix', 
                           help='选择使用的模型类型 [cycle_gan | pix2pix | test | colorization]')
        
        parser.add_argument('--input_nc', type=int, default=1, 
                           help='输入图像通道数: RGB图像为3，灰度图像为1')
        
        parser.add_argument('--output_nc', type=int, default=1, 
                           help='输出图像通道数: RGB图像为3，灰度图像为1')
        
        # ===== 生成器网络参数 =====
        parser.add_argument('--ngf', type=int, default=64, 
                           help='生成器最后一层卷积层的滤波器数量')
        
        parser.add_argument('--netG', type=str, default='unet_128', 
                           help='生成器架构 [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        
        # ===== 判别器网络参数 =====
        parser.add_argument('--ndf', type=int, default=64, 
                           help='判别器第一层卷积层的滤波器数量')
        
        parser.add_argument('--netD', type=str, default='basic', 
                           help='判别器架构 [basic | n_layers | pixel]。basic是70x70 PatchGAN')
        
        parser.add_argument('--n_layers_D', type=int, default=3, 
                           help='判别器层数，仅当netD==n_layers时使用')
        
        # ===== 网络训练参数 =====
        parser.add_argument('--norm', type=str, default='instance', 
                           help='归一化方法 [instance | batch | none]')
        
        parser.add_argument('--init_type', type=str, default='normal', 
                           help='网络初始化方法 [normal | xavier | kaiming | orthogonal]')
        
        parser.add_argument('--init_gain', type=float, default=0.02, 
                           help='normal、xavier、orthogonal初始化的缩放因子')
        
        parser.add_argument('--no_dropout', action='store_true', 
                           help='生成器不使用dropout')
        
        parser.add_argument('--use_sab', action='store_true', 
                           help='使用自注意力块(Self-Attention Block)')
        
        # ===== 训练配置参数 =====
        parser.add_argument('--batch_size', type=int, default=1, 
                           help='输入批次大小')
        
        parser.add_argument('--epoch', type=str, default='latest', 
                           help='加载哪个epoch的模型？设为latest使用最新缓存模型')
        
        parser.add_argument('--load_iter', type=int, default='0', 
                           help='加载哪个iteration的模型？如果>0，按iter_[load_iter]加载；否则按[epoch]加载')
        
        # ===== 系统参数 =====
        parser.add_argument('--verbose', action='store_true', 
                           help='打印更多调试信息')
        
        parser.add_argument('--suffix', default='', type=str, 
                           help='自定义后缀: opt.name = opt.name + suffix，例如: {model}_{netG}_size{load_size}')
        
        self.initialized = True
        return parser

    def gather_options(self):
        """
        收集并整理所有配置选项
        
        这个方法执行配置收集的完整流程：
        1. 初始化基础配置解析器
        2. 获取模型名称
        3. 加载模型特定的配置选项
        4. 重新解析以包含所有选项
        
        返回：
        解析后的配置选项对象
        """
        # ===== 初始化解析器（仅执行一次）=====
        if not self.initialized:
            # 创建参数解析器，使用默认值格式化器显示帮助信息
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # ===== 获取基础选项 =====
        # parse_known_args()返回已知参数和未知参数，忽略未知参数
        opt, _ = parser.parse_known_args()

        # ===== 添加模型特定的选项 =====
        # 根据模型类型获取对应的选项设置器
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        # 调用模型的选项设置器，添加模型特定参数
        parser = model_option_setter(parser, self.isTrain)
        
        # 重新解析以包含新添加的模型特定选项
        opt, _ = parser.parse_known_args()

        # ===== 保存解析器并返回最终选项 =====
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """
        打印和保存配置选项
        
        功能：
        1. 格式化显示所有配置参数
        2. 标出与默认值不同的参数
        3. 将配置保存到文件
        
        参数：
        opt: 配置选项对象
        """
        
        # ===== 构建显示消息 =====
        message = ''
        message += '----------------- Options ---------------\n'
        
        # 遍历所有配置选项，按字母顺序排序
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            
            # 如果当前值与默认值不同，显示默认值
            if v != default:
                comment = '\t[default: %s]' % str(default)
                
            # 格式化输出：参数名(25字符右对齐) : 参数值(30字符左对齐) 默认值注释
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
            
        message += '----------------- End -------------------'
        
        # ===== 打印到控制台 =====
        print(message)

        # ===== 保存到文件 =====
        # 创建实验目录
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)  # 创建目录（如果不存在）
        
        # 保存配置到opt.txt文件
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """
        解析配置选项，创建检查点目录，设置GPU设备
        
        这是配置系统的主入口方法，执行完整的配置流程：
        1. 收集所有配置选项
        2. 处理实验名称后缀
        3. 打印和保存配置
        4. 设置GPU设备
        
        返回：
        完全配置好的选项对象
        """
        
        # ===== 收集配置选项 =====
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # 设置训练/测试标志

        # ===== 处理实验名称后缀 =====
        if opt.suffix:
            # 使用配置参数格式化后缀字符串
            # 例如：suffix="{model}_{netG}" -> "pix2pix_unet_256"
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # ===== 打印配置信息 =====
        self.print_options(opt)

        # ===== 设置GPU设备 =====
        # 解析GPU ID字符串，支持多GPU
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:  # 有效的GPU ID
                opt.gpu_ids.append(id)
                
        # 如果有可用GPU，设置主GPU设备
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # ===== 保存并返回配置 =====
        self.opt = opt
        return self.opt


# ===== 使用说明和架构解析 =====
"""
BaseOptions在AUGAN项目中的作用：

1. 配置管理中心：
   - 统一管理所有训练和测试参数
   - 提供命令行接口和程序接口
   - 确保配置的一致性和可重现性

2. 模型配置：
   - 网络架构选择(U-Net, ResNet)
   - 网络参数设置(通道数, 滤波器数)
   - 训练策略配置(归一化, 初始化)

3. 实验管理：
   - 实验命名和组织
   - 检查点保存和加载
   - 结果目录管理

4. 系统配置：
   - GPU设备管理
   - 批次大小设置
   - 调试信息控制

5. 扩展机制：
   - 支持模型特定配置
   - 支持自定义后缀
   - 支持配置文件保存

典型使用方式：
```python
# 训练时使用
from options.train_options import TrainOptions
opt = TrainOptions().parse()

# 测试时使用  
from options.test_options import TestOptions
opt = TestOptions().parse()

# 命令行使用
python train.py --name unet_exp --netG unet_256 --batch_size 4
```

配置继承结构：
BaseOptions (基础配置)
├── TrainOptions (添加训练特定配置)
│   ├── 学习率、损失函数
│   ├── 数据增强、优化器
│   └── 训练轮数、保存频率
└── TestOptions (添加测试特定配置)
    ├── 测试数据路径
    ├── 结果保存选项
    └── 评估指标设置

重要配置参数解释：
- name: 实验名称，影响保存路径
- netG/netD: 生成器/判别器架构
- input_nc/output_nc: 输入/输出通道数
- batch_size: 批次大小，影响显存使用
- gpu_ids: GPU设备，支持多卡训练
- checkpoints_dir: 模型保存目录
"""