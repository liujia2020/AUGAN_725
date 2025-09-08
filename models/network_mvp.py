from __future__ import print_function
#!/usr/bin/env python3


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

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):

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

    net = None
    norm_layer = get_norm_layer(norm_type=norm)


    if netG == 'unet_128':
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
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', 
             init_type='normal', init_gain=0.02, gpu_ids=[]):

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        # 默认的PatchGAN判别器 (70×70感受野)
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':
        # 可配置层数的PatchGAN判别器
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    
    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):

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

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        
        # expand_as: 将标量扩展为与prediction相同的形状
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):

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

    def __init__(self, cnn, feature_layer=11):

        super(FeatureExtractor, self).__init__()
        # 只保留到指定层的网络结构
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        """提取特征"""
        return self.features(x)

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


    def __init__(self, input_nc, 
                 output_nc, 
                 num_downs, 
                 ngf=64, 
                 norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, 
                 use_sab=False):

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

        if self.outermost:
            # 最外层: 直接返回输出，无跳跃连接
            return self.model(x)
        else:
            if self.inter == False:
                # 非中间层: 使用像素注意力增强跳跃连接
                x2 = self.pa(x)           # 计算注意力权重
                x3 = torch.mul(x2, x)     # 注意力调制输入特征
                # 跳跃连接: 拼接调制后的输入和处理后的特征，在通道维度直接相加
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

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):

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

        return self.model(input)


