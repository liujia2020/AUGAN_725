#!/usr/bin/env python3
"""
AUGAN 网络架构模块详细注释版本
network.py - 定义生成器、判别器、损失函数等核心网络组件
包含 U-Net 生成器、PatchGAN 判别器、各种注意力机制的完整实现
"""

# ==================== 网络架构总结 ====================
"""
network.py 核心组件总结:

1. **生成器架构**:
   - UnetGenerator: AUGAN使用，适合像素级任务
   - ResnetGenerator: 基于残差网络，适合全局变换

2. **判别器架构**:
   - NLayerDiscriminator: PatchGAN判别器，判断局部真实性
   - PixelDiscriminator: 像素级判别器

3. **损失函数**:
   - GANLoss: 支持多种GAN损失 (lsgan, vanilla, wgangp)
   - FeatureExtractor: 用于感知损失计算

4. **初始化工具**:
   - init_weights: 网络权重初始化
   - init_net: 网络设备设置和初始化
   - get_norm_layer: 规范化层选择
   - get_scheduler: 学习率调度器

5. **AUGAN配置**:
   - 生成器: UnetGenerator (8层, 256×256)
   - 判别器: NLayerDiscriminator (3层PatchGAN)
   - 损失: lsgan损失
   - 规范化: InstanceNorm
   - 初始化: normal (0.02)

这个模块提供了构建GAN网络的所有基础组件，
AUGAN通过组合这些组件实现超声图像增强功能。
"""

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import copy
from models.SAB import ChannelAttention, SpatialAttention, SpatioAttention, LocalAwareAttention, GlobalAwareAttention, PixelAwareAttention


def get_norm_layer(norm_type='instance'):
    """
    获取规范化层
    
    参数:
        norm_type (str): 规范化类型 - 'batch' | 'instance' | 'none'
    
    返回:
        norm_layer: 对应的规范化层函数
    
    规范化层说明:
        - BatchNorm: 对每个batch进行规范化，适用于batch size较大的情况
        - InstanceNorm: 对每个样本单独规范化，适用于风格迁移等任务
        - None: 不使用规范化层
    
    在AUGAN中的应用:
        主要使用 InstanceNorm，因为超声图像增强是图像到图像的翻译任务
        InstanceNorm 能更好地保持图像的风格和细节特征
    """
    if norm_type == 'batch':
        # BatchNorm2d: 在batch维度上计算均值和方差
        # affine=True: 学习缩放和偏移参数
        # track_running_stats=True: 跟踪运行时统计信息
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        
    elif norm_type == 'instance':
        # InstanceNorm2d: 对每个样本的每个通道单独规范化
        # affine=False: 不学习额外的缩放和偏移参数
        # track_running_stats=False: 不跟踪运行时统计
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        
    elif norm_type == 'none':
        # 不使用规范化层
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """
    获取学习率调度器
    
    参数:
        optimizer: 优化器对象
        opt: 配置选项对象，包含学习率策略参数
    
    返回:
        scheduler: 对应的学习率调度器
    
    支持的学习率策略:
        - linear: 前期保持不变，后期线性衰减到0
        - step: 每隔固定步数减少学习率
        - plateau: 根据监控指标自适应调整
        - cosine: 余弦退火调度
    
    AUGAN中的应用:
        通常使用linear策略，前期稳定训练，后期逐步降低学习率保证收敛
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            """
            线性学习率衰减函数
            
            前 opt.niter 个epoch保持原始学习率
            后续 opt.niter_decay 个epoch线性衰减到0
            """
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        
    elif opt.lr_policy == 'step':
        # 每隔 lr_decay_iters 步，学习率乘以0.1
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        
    elif opt.lr_policy == 'plateau':
        # 当监控指标停止改善时，降低学习率
        # mode='min': 监控指标越小越好
        # factor=0.2: 每次减少到原来的20%
        # patience=5: 连续5个epoch无改善才调整
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, 
                                                 threshold=0.01, patience=5)
        
    elif opt.lr_policy == 'cosine':
        # 余弦退火：学习率按余弦函数变化
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    初始化网络权重
    
    参数:
        net: 要初始化的网络
        init_type (str): 初始化方法 - 'normal' | 'xavier' | 'kaiming' | 'orthogonal'
        init_gain (float): 初始化缩放因子
    
    初始化方法说明:
        - normal: 正态分布初始化，均值0，标准差init_gain
        - xavier: Xavier初始化，适用于sigmoid/tanh激活函数
        - kaiming: He初始化，适用于ReLU激活函数
        - orthogonal: 正交初始化，保持梯度流稳定
    
    在AUGAN中的应用:
        通常使用normal初始化，init_gain=0.02，适合GAN网络的稳定训练
    """
    def init_func(m):
        classname = m.__class__.__name__
        
        # 对卷积层和线性层进行权重初始化
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                # 正态分布初始化: N(0, init_gain²)
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                # Xavier正态分布初始化
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                # Kaiming正态分布初始化 (He初始化)
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                # 正交初始化
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            
            # 如果有偏置项，初始化为0
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
                
        # 对BatchNorm层进行特殊初始化
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm的权重初始化为正态分布: N(1.0, init_gain²)
            init.normal_(m.weight.data, 1.0, init_gain)
            # BatchNorm的偏置初始化为0
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # 递归应用初始化函数到所有子模块


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    初始化网络: 设置设备 + 初始化权重
    
    参数:
        net: 网络对象
        init_type (str): 权重初始化方法
        init_gain (float): 初始化缩放因子
        gpu_ids (list): 使用的GPU ID列表
    
    返回:
        初始化完成的网络
    
    功能:
        1. 将网络移动到指定设备(GPU/CPU)
        2. 设置多GPU并行 (如果有多个GPU)
        3. 初始化网络权重
    """
    if len(gpu_ids) > 0:
        # 确保CUDA可用
        assert(torch.cuda.is_available())
        # 将网络移动到第一个GPU
        net.to(gpu_ids[0])
        # 设置多GPU并行训练
        net = torch.nn.DataParallel(net, gpu_ids)
    
    # 初始化网络权重
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, 
             init_type='normal', init_gain=0.02, gpu_ids=[], use_sab=False):
    """
    创建生成器网络
    
    参数:
        input_nc (int): 输入图像通道数
        output_nc (int): 输出图像通道数
        ngf (int): 生成器最后一层卷积的滤波器数量
        netG (str): 生成器架构名称
        norm (str): 规范化层类型
        use_dropout (bool): 是否使用dropout
        init_type (str): 权重初始化方法
        init_gain (float): 初始化缩放因子
        gpu_ids (list): GPU设备列表
        use_sab (bool): 是否使用自注意力机制
    
    返回:
        初始化完成的生成器网络
    
    支持的生成器架构:
        - resnet_9blocks/resnet_6blocks: ResNet生成器
        - unet_128/unet_256: U-Net生成器 (AUGAN使用)
        - srgan: 超分辨率生成器
        - VDSR: 深度超分辨率网络
    
    AUGAN中的配置:
        通常使用 unet_256，即8层下采样的U-Net架构
        输入输出都是单通道(灰度图像)，ngf=64
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        # 9个ResNet块的生成器
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, 
                             use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        # 6个ResNet块的生成器
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, 
                             use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        # 适用于128×128图像的U-Net (7层下采样)
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, 
                           use_dropout=use_dropout, use_sab=use_sab)
    elif netG == 'unet_256':
        # 适用于256×256图像的U-Net (8层下采样) - AUGAN使用
        net = UnetGenerator(input_nc, 
                            output_nc, 
                            8, 
                            ngf, 
                            norm_layer=norm_layer, 
                            use_dropout=use_dropout, 
                            use_sab=use_sab)
    elif netG == 'srgan':
        # 超分辨率GAN生成器
        net = SRGenerator()
    elif netG == 'VDSR':
        # 深度超分辨率网络
        net = VDCGAN()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', 
             init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    创建判别器网络
    
    参数:
        input_nc (int): 输入图像通道数
        ndf (int): 判别器第一层卷积的滤波器数量
        netD (str): 判别器架构名称
        n_layers_D (int): 判别器卷积层数量
        norm (str): 规范化层类型
        init_type (str): 权重初始化方法
        init_gain (float): 初始化缩放因子
        gpu_ids (list): GPU设备列表
    
    返回:
        初始化完成的判别器网络
    
    支持的判别器类型:
        - basic: 标准PatchGAN判别器 (70×70 patch)
        - n_layers: 可指定层数的PatchGAN判别器
        - pixel: 1×1像素级判别器
        - srgan: 超分辨率判别器
    
    PatchGAN原理:
        不是判别整张图像的真假，而是判断图像中每个patch的真假
        70×70 PatchGAN能够捕获局部纹理特征，参数更少，训练更稳定
    
    AUGAN中的配置:
        使用 basic PatchGAN，即3层卷积的70×70感受野判别器
        适合判别超声图像的局部纹理真实性
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        # 默认的PatchGAN判别器 (70×70感受野)
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':
        # 可配置层数的PatchGAN判别器
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':
        # 1×1像素级判别器
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'srgan':
        # 超分辨率判别器
        net = SRDiscriminator()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    
    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    """
    GAN损失函数类
    
    功能:
        封装不同类型的GAN损失函数
        自动处理真假标签的生成
        支持多种GAN训练目标
    
    支持的GAN损失类型:
        - vanilla: 原始GAN损失 (使用BCEWithLogitsLoss)
        - lsgan: 最小二乘GAN损失 (使用MSELoss)
        - wgangp: Wasserstein GAN with Gradient Penalty
    
    在AUGAN中的应用:
        通常使用lsgan损失，训练更稳定，生成质量更好
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """
        初始化GAN损失函数
        
        参数:
            gan_mode (str): GAN损失类型
            target_real_label (float): 真实样本的标签值
            target_fake_label (float): 生成样本的标签值
        """
        super(GANLoss, self).__init__()
        
        # 注册为buffer，不会被优化器更新，但会随模型移动设备
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            # 最小二乘GAN: 使用L2损失，训练更稳定
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            # 原始GAN: 使用交叉熵损失
            # BCEWithLogitsLoss = Sigmoid + BCELoss，数值更稳定
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            # Wasserstein GAN: 不需要损失函数，直接计算Wasserstein距离
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """
        创建与预测张量相同形状的标签张量
        
        参数:
            prediction (tensor): 判别器的预测输出
            target_is_real (bool): 是否为真实样本
        
        返回:
            与prediction形状相同的标签张量
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        
        # expand_as: 将标量扩展为与prediction相同的形状
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """
        计算GAN损失
        
        参数:
            prediction (tensor): 判别器的预测输出
            target_is_real (bool): 是否为真实样本
        
        返回:
            计算得到的损失值
        
        不同损失的计算方式:
            - lsgan/vanilla: 使用对应的损失函数
            - wgangp: 直接计算均值 (真实样本取负值，生成样本取正值)
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            # 生成标签张量
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            # 计算损失
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            # Wasserstein GAN损失
            if target_is_real:
                # 真实样本: 最大化判别器输出 (最小化负值)
                loss = -prediction.mean()
            else:
                # 生成样本: 最小化判别器输出
                loss = prediction.mean()
        return loss




class FeatureExtractor(nn.Module):
    """
    特征提取器
    
    功能:
        从预训练CNN中提取中间层特征
        用于感知损失计算或特征匹配
    
    应用场景:
        - 计算感知损失 (perceptual loss)
        - 特征匹配损失
        - 风格损失计算
    """
    
    def __init__(self, cnn, feature_layer=11):
        """
        初始化特征提取器
        
        参数:
            cnn: 预训练的CNN网络 (如VGG)
            feature_layer (int): 提取第几层的特征
        """
        super(FeatureExtractor, self).__init__()
        # 只保留到指定层的网络结构
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        """提取特征"""
        return self.features(x)


class ResnetGenerator(nn.Module):
    """
    基于ResNet的生成器
    
    网络结构:
        输入 → 编码器(下采样) → ResNet块 → 解码器(上采样) → 输出
    
    特点:
        - 使用残差连接避免梯度消失
        - 对称的编码-解码结构
        - 适用于需要保持细节的图像翻译任务
    
    在AUGAN中的应用:
        虽然定义了ResNet生成器，但AUGAN主要使用U-Net架构
        U-Net的跳跃连接更适合超声图像增强任务
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, n_blocks=6, padding_type='reflect'):
        """
        构造ResNet生成器
        
        参数:
            input_nc (int): 输入通道数
            output_nc (int): 输出通道数
            ngf (int): 最后一层卷积的滤波器数量
            norm_layer: 规范化层
            use_dropout (bool): 是否使用dropout
            n_blocks (int): ResNet块的数量
            padding_type (str): 填充类型 'reflect' | 'replicate' | 'zero'
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        
        # 判断是否使用偏置
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 网络结构列表
        model = [
            nn.ReflectionPad2d(3),  # 反射填充，减少边界伪影
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        # 编码器: 下采样层
        n_downsampling = 3
        for i in range(n_downsampling):
            mult = 2 ** i  # 通道数倍增: 64 → 128 → 256 → 512
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, 
                         padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # ResNet块: 特征变换
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, 
                                norm_layer=norm_layer, use_dropout=use_dropout, 
                                use_bias=use_bias)]

        # 解码器: 上采样层
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                 kernel_size=3, stride=2,
                                 padding=1, output_padding=1,
                                 bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        
        # 输出层
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # 注意: 没有使用Tanh激活，输出值域不限制在[-1,1]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """前向传播"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """
    ResNet残差块
    
    结构:
        输入 → 卷积块 → 输出 + 输入 (残差连接)
    
    功能:
        - 残差连接解决梯度消失问题
        - 使网络能够训练更深
        - 保持特征信息流
    
    AUGAN中的应用:
        在ResNet生成器中使用，但AUGAN主要使用U-Net
        ResNet块中集成了LocalAwareAttention注意力机制
    """

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        初始化ResNet块
        
        参数:
            dim (int): 通道数
            padding_type (str): 填充类型
            norm_layer: 规范化层
            use_dropout (bool): 是否使用dropout
            use_bias (bool): 是否使用偏置
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, 
                                              use_dropout, use_bias)
        # 集成局部感知注意力机制
        self.la = LocalAwareAttention()

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        构建卷积块
        
        结构:
            填充 → 卷积 → 规范化 → ReLU → [Dropout] → 填充 → 卷积 → 规范化
        
        返回:
            卷积块的Sequential模块
        """
        conv_block = []
        
        # 第一个卷积层的填充
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        # 第一个卷积 + 规范化 + 激活
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
            norm_layer(dim), 
            nn.ReLU(True)
        ]
        
        # 可选的dropout
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        # 第二个卷积层的填充
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
        # 第二个卷积 + 规范化 (不加激活函数)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """
        前向传播 (带残差连接)
        
        计算流程:
            1. 输入通过卷积块处理
            2. 与原始输入相加 (残差连接)
            3. 输出结果
        """
        x1 = self.conv_block(x)
        out = x + x1  # 残差连接
        return out

'''
1️⃣ 生成器 (Generator):
  类名: UnetGenerator
  位置: models/network. Py 第 385 行
  作用: 单角度图像 → 高质量图像
  架构: U-Net (编码器-解码器+跳跃连接)
  输入: real_A (1, 512, 384)
  输出: fake_B (1, 512, 384)
  为什么选择: U-Net 适合像素级图像翻译，保持细节
  '''
class UnetGenerator(nn.Module):
    """
    U-Net生成器 - AUGAN的核心网络架构
    
    网络结构:
        编码器 (下采样) ← → 解码器 (上采样)
                  ↓         ↑
                瓶颈层 (最深层)
    
    关键特性:
        1. 跳跃连接: 直接连接编码器和解码器的对应层
        2. 对称结构: 编码器和解码器层数相同
        3. 注意力机制: 集成多种注意力模块增强性能
        4. 多尺度特征: 不同层捕获不同尺度的特征
    
    在AUGAN中的应用:
        - 输入: 单角度超声重建图像 (1通道)
        - 输出: 增强后的超声图像 (1通道)
        - 尺寸: 256×256 (使用unet_256配置)
        - 层数: 8层下采样 (num_downs=8)
    
    优势:
        - 跳跃连接保持细节信息
        - 适合像素级预测任务
        - 能够恢复高频细节
        - 训练相对稳定
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, use_sab=False):
        """
        构造U-Net生成器
        
        参数:
            input_nc (int): 输入通道数 (AUGAN中为1)
            output_nc (int): 输出通道数 (AUGAN中为1)
            num_downs (int): 下采样层数 (AUGAN中为8)
            ngf (int): 最后一层卷积的滤波器数 (通常64)
            norm_layer: 规范化层 (通常InstanceNorm)
            use_dropout (bool): 是否使用dropout
            use_sab (bool): 是否使用自注意力模块
        
        构造方式:
            从内层到外层递归构造，每个UnetSkipConnectionBlock
            包含一个编码-解码对，并包含到子模块的跳跃连接
        """
        super(UnetGenerator, self).__init__()
        
        # 从最内层开始构造 (瓶颈层)
        # 最内层: ngf*8 → ngf*8, 没有子模块
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, 
            ngf * 8, 
            input_nc=None, 
            submodule=None, 
            norm_layer=norm_layer, 
            innermost=True, 
            inter=True
        )
        
        # 判断偏置使用
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 中间层: 都是 ngf*8 → ngf*8 的结构
        # 对于256×256图像 (num_downs=8)，会有3个中间层
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, 
                ngf * 8, 
                input_nc=None, 
                submodule=unet_block, 
                norm_layer=norm_layer, 
                use_dropout=use_dropout, 
                inter=True
            )

        # 逐渐减少滤波器数量的层
        # ngf*4 → ngf*8  (对应32×24分辨率)
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, 
            ngf * 8, 
            input_nc=None, 
            submodule=unet_block, 
            norm_layer=norm_layer
        )
        
        # ngf*2 → ngf*4  (对应64×48分辨率)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, 
            ngf * 4, 
            input_nc=None, 
            submodule=unet_block, 
            norm_layer=norm_layer
        )
        
        # ngf → ngf*2   (对应128×96分辨率)
        unet_block = UnetSkipConnectionBlock(
            ngf, 
            ngf * 2, 
            input_nc=None, 
            submodule=unet_block, 
            norm_layer=norm_layer
        )
        
        # 最外层: output_nc → ngf  (对应256×192分辨率)
        self.model = UnetSkipConnectionBlock(
            output_nc, 
            ngf, 
            input_nc=input_nc, 
            submodule=unet_block, 
            outermost=True, 
            norm_layer=norm_layer
        )

    def forward(self, input):
        """
        前向传播
        
        数据流:
            输入图像 → 最外层UnetBlock → 递归处理所有层 → 输出图像
        
        在每个UnetBlock中:
            1. 下采样 (编码)
            2. 传递给子模块
            3. 上采样 (解码)
            4. 与编码特征拼接 (跳跃连接)
        """
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """
    U-Net跳跃连接模块 - U-Net的基本构建单元
    
    设计理念:
        每个模块包含一个编码-解码对，通过跳跃连接保持细节信息
        递归嵌套结构，从内层到外层逐步构建完整的U-Net
    
    网络结构:
        输入 → 下采样(编码) → 子模块 → 上采样(解码) → 跳跃连接拼接 → 输出
    
    三种模块类型:
        1. outermost: 最外层，没有规范化，处理原始输入输出
        2. innermost: 最内层，没有子模块，作为递归终止
        3. intermediate: 中间层，标准编码-解码结构
    
    在AUGAN中的创新:
        - 集成PixelAwareAttention注意力机制
        - 可选的LocalAwareAttention (已注释)
        - inter参数控制是否使用注意力
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, 
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, inter=False):
        """
        构造U-Net跳跃连接模块
        
        参数:
            outer_nc (int): 外层（输出）通道数
            inner_nc (int): 内层（中间）通道数
            input_nc (int): 输入通道数，如果为None则等于outer_nc
            submodule: 子模块，递归嵌套的下一层
            outermost (bool): 是否为最外层
            innermost (bool): 是否为最内层  
            norm_layer: 规范化层类型
            use_dropout (bool): 是否使用dropout
            inter (bool): 是否为中间层(影响注意力机制使用)
        
        模块构建逻辑:
            1. 最外层: 输入→编码→子模块→解码→输出 (无跳跃连接)
            2. 最内层: 输入→编码→解码→输出 (递归终止)
            3. 中间层: 输入→编码→子模块→解码→与输入拼接 (跳跃连接)
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.innermost = innermost
        self.outermost = outermost
        self.inter = inter  # 中间层标志，用于控制注意力机制

        # 判断是否使用偏置
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        # 设置输入通道数
        if input_nc is None:
            input_nc = outer_nc
        self.input_nc = input_nc
        self.outer_nc = outer_nc

        # 定义基本层组件
        # 下采样卷积: 4×4卷积，步长2，输出尺寸减半
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, False)  # 下采样激活
        downnorm = norm_layer(inner_nc)      # 下采样规范化
        uprelu = nn.LeakyReLU(0.2, False)    # 上采样激活
        upnorm = norm_layer(outer_nc)        # 上采样规范化
        
        # 像素感知注意力机制
        self.pixelatt = PixelAwareAttention(inner_nc)

        if outermost:
            """
            最外层模块:
            - 输入: 原始图像
            - 输出: 生成图像
            - 特点: 无规范化，通道翻倍拼接，无Tanh激活
            """
            # 上采样: inner_nc*2 → outer_nc (因为有跳跃连接拼接)
            upconv = nn.ConvTranspose2d(inner_nc * 2, 
                                        outer_nc, 
                                        kernel_size=4, 
                                        stride=2, 
                                        padding=1)
            down = [downconv]  # 只有下采样卷积，无规范化和激活
            up = [upconv, uprelu]  # 上采样 + 激活，无Tanh
            model = down + [submodule] + up

        elif innermost:
            """
            最内层模块:
            - 特点: 最深层，无子模块，递归终止
            - 功能: 特征压缩到最小空间分辨率
            - 通道: inner_nc → outer_nc (无拼接)
            """
            # 上采样: inner_nc → outer_nc (无跳跃连接)
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, 
                                      stride=2, padding=1, bias=use_bias)
            down = [downconv, downrelu]  # 下采样 + 激活
            
            if self.inter == True:
                # 中间层配置: 添加规范化
                up = [upconv, uprelu, upnorm]
            else:
                # 标准配置: 添加规范化
                up = [upconv, uprelu, upnorm]
            
            model = down + up  # 无子模块

        else:
            """
            中间层模块:
            - 特点: 标准编码-解码结构
            - 功能: 特征变换和传递
            - 通道: inner_nc*2 → outer_nc (跳跃连接拼接)
            """
            # 上采样: inner_nc*2 → outer_nc (考虑跳跃连接)
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, 
                                      stride=2, padding=1, bias=use_bias)
            down = [downconv, downrelu, downnorm]  # 完整下采样流程
            
            if self.inter == True:
                # 中间层: 标准上采样
                up = [upconv, uprelu, upnorm]
            else:
                # 标准配置: 标准上采样
                up = [upconv, uprelu, upnorm]

            # 可选的dropout
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        
        # 像素感知注意力 (用于非中间层)
        self.pa = PixelAwareAttention(input_nc)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征图
        
        返回:
            处理后的特征图
        
        数据流程:
            1. 最外层: 直接返回模型输出 (无跳跃连接)
            2. 其他层: 将输入与模型输出在通道维度拼接 (跳跃连接)
        
        注意力机制:
            - inter=False时: 使用PixelAwareAttention调制输入
            - inter=True时: 标准跳跃连接，无注意力
        """
        if self.outermost:
            # 最外层: 直接返回输出，无跳跃连接
            return self.model(x)
        else:
            if self.inter == False:
                # 非中间层: 使用像素注意力增强跳跃连接
                x2 = self.pa(x)           # 计算注意力权重
                x3 = torch.mul(x2, x)     # 注意力调制输入特征
                # 跳跃连接: 拼接调制后的输入和处理后的特征
                return torch.cat([x3, self.model(x)], 1)
            else:
                # 中间层: 标准跳跃连接
                return torch.cat([x, self.model(x)], 1)




'''
2️⃣ 判别器 (Discriminator):
  类名: NLayerDiscriminator
  位置: models/network. Py 第 555 行
  作用: 判断图像对的真假
  架构: PatchGAN (70×70 感受野)
  输入: 图像对 (2, 512, 384)
  输出: 预测图 (1, 32, 24)
  为什么选择: PatchGAN 关注局部细节，参数少
'''
class NLayerDiscriminator(nn.Module):
    """
    PatchGAN判别器 - AUGAN的判别器网络
    
    设计原理:
        不判别整张图像真假，而是判别图像中每个patch的真假
        70×70感受野的patch能捕获足够的局部纹理信息
        参数更少，训练更稳定，适合高分辨率图像
    
    网络结构:
        输入 → 卷积层1 → 卷积层2 → ... → 卷积层N → 1×1输出层
        每层步长为2，逐步降低空间分辨率
        最后输出每个patch的真假概率图
    
    在AUGAN中的应用:
        - 输入: 单角度+多角度复合图像 (2通道)
        - 输出: 每个patch的真假判别
        - 感受野: 70×70像素 (n_layers=3)
        - 有效判别超声图像的局部纹理真实性
    
    优势:
        - 局部判别，关注纹理细节
        - 参数效率高
        - 训练稳定
        - 适合高分辨率图像
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        构造PatchGAN判别器
        
        参数:
            input_nc (int): 输入通道数 (AUGAN中为2: 单角度+多角度)
            ndf (int): 第一层卷积滤波器数量 (默认64)
            n_layers (int): 卷积层数量 (默认3，对应70×70感受野)
            norm_layer: 规范化层类型
        
        网络配置:
            - Layer 1: input_nc → ndf,     stride=2, 无规范化
            - Layer 2: ndf → ndf*2,       stride=2, 有规范化  
            - Layer 3: ndf*2 → ndf*4,     stride=2, 有规范化
            - Layer 4: ndf*4 → ndf*8,     stride=1, 有规范化
            - Output:  ndf*8 → 1,         stride=1, 输出概率图
        """
        super(NLayerDiscriminator, self).__init__()
        
        # 判断是否使用偏置
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4      # 卷积核大小
        padw = 1    # 填充大小
        
        # 第一层: 输入层，无规范化
        # input_nc → ndf, stride=2, 空间尺寸减半
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1      # 当前层的通道倍数
        nf_mult_prev = 1 # 前一层的通道倍数
        
        # 中间层: 逐步增加通道数，减少空间尺寸
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # 通道倍数最大为8
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                         kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # 倒数第二层: stride=1，保持空间尺寸
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                     kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # 输出层: 生成单通道概率图
        # ndf*nf_mult → 1, stride=1, 每个像素对应一个patch的判别结果
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """
        前向传播
        
        参数:
            input: 输入图像 (B, C, H, W)
                  - B: batch size
                  - C: 通道数 (AUGAN中为2)
                  - H, W: 图像尺寸
        
        返回:
            patch判别结果 (B, 1, H', W')
            - H', W': 经过下采样后的尺寸
            - 每个像素值表示对应patch为真实的概率
        
        感受野计算:
            对于3层网络 (n_layers=3):
            - 每层stride=2的4×4卷积
            - 最终感受野约为70×70像素
        """
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """
    像素级判别器 (1×1 PatchGAN)
    
    设计原理:
        每个像素独立判别真假，感受野为1×1
        主要关注颜色和亮度的合理性，不考虑空间结构
        适合需要精确像素级控制的任务
    
    网络结构:
        简单的三层1×1卷积网络
        不改变空间分辨率，只变换通道数
    
    应用场景:
        - 颜色一致性约束
        - 像素级真实性判别
        - 轻量级判别器需求
    
    在AUGAN中的地位:
        虽然定义了PixelDiscriminator，但AUGAN主要使用PatchGAN
        PatchGAN的局部判别更适合超声图像纹理特征
    """

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """
        构造1×1像素判别器
        
        参数:
            input_nc (int): 输入通道数
            ndf (int): 中间层滤波器数量
            norm_layer: 规范化层类型
        
        网络结构:
            input_nc → ndf → ndf*2 → 1
            所有卷积都是1×1，保持空间分辨率不变
        """
        super(PixelDiscriminator, self).__init__()
        
        # 判断是否使用偏置
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            # 第一层: input_nc → ndf
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            
            # 第二层: ndf → ndf*2
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            
            # 输出层: ndf*2 → 1
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """
        前向传播
        
        参数:
            input: 输入图像 (B, C, H, W)
        
        返回:
            像素级判别结果 (B, 1, H, W)
            - 空间分辨率与输入相同
            - 每个像素独立判别真假
        """
        return self.net(input)


