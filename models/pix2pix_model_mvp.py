#!/usr/bin/env python3
"""
AUGAN Pix2Pix模型详细注释版本
pix2pix_model.py - AUGAN的核心训练模型，实现条件GAN的完整训练逻辑
基于Pix2Pix框架，专门针对超声图像增强任务进行了优化
"""

import torch
from .base_model import BaseModel
# from . import network
import models.network_mvp as network
import torchvision
from thop import profile


class Pix2PixModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # 设置Pix2Pix的默认值，匹配论文配置
        parser.set_defaults(
            norm='instance',     # 实例规范化，适合风格迁移
            netG='unet_128',    # U-Net生成器 (AUGAN实际使用unet_256)
            netD='basic',       # 基础PatchGAN判别器
            use_sab=False,      # 不使用自注意力机制
            name='unet_b002'    # 实验名称
        )
        
        if is_train:
            # 训练模式特定参数
            parser.set_defaults(
                pool_size=0,        # 不使用图像池 (Pix2Pix不需要)
                gan_mode='vanilla'   # 使用原始GAN损失
            )
            # 添加L1损失权重参数
            parser.add_argument('--lambda_L1', type=float, default=1, 
                              help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        # 定义要打印的损失名称
        # 训练脚本会调用get_current_losses()获取这些损失值
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        
        # 创建VGG特征提取器用于感知损失
        # 使用预训练的VGG19网络提取高级特征
        vgg = torchvision.models.vgg19(pretrained=True)
        # 修改第一层以接受单通道输入 (原本是3通道RGB)
        vgg.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # 创建特征提取器，提取第11层的特征用于感知损失
        self.feature_extractor = network.FeatureExtractor(vgg)
        
        # 定义要保存的模型名称
        if self.isTrain:
            # 训练模式: 保存生成器和判别器
            self.model_names = ['G', 'D']
        else:
            # 测试模式: 只需要生成器
            self.model_names = ['G']
            
        # 定义生成器网络
        # define_G函数会根据opt参数创建对应的网络架构
        self.netG = network.define_G(
            opt.input_nc,        # 输入通道数 (AUGAN中为1)
            opt.output_nc,       # 输出通道数 (AUGAN中为1)
            opt.ngf,             # 生成器滤波器数量 (通常64)
            opt.netG,            # 生成器架构 (unet_256)
            opt.norm,            # 规范化类型 (instance)
            not opt.no_dropout,  # 是否使用dropout
            opt.init_type,       # 权重初始化方法
            opt.init_gain,       # 初始化缩放因子
            self.gpu_ids,        # GPU设备列表
            opt.use_sab          # 是否使用自注意力
        )

        if self.isTrain:
            # 训练模式: 定义判别器
            # 条件GAN需要同时输入原图和目标图，所以通道数是两者之和 [模糊图像, 清晰图像]拼接在一起（2个通道）
            self.netD = network.define_D(
                opt.input_nc + opt.output_nc,  # 输入通道数 (AUGAN中为2)
                opt.ndf,                       # 判别器滤波器数量 (通常64)
                opt.netD,                      # 判别器架构 (basic)
                opt.n_layers_D,                # 判别器层数 (通常3)
                opt.norm,                      # 规范化类型
                opt.init_type,                 # 权重初始化方法
                opt.init_gain,                 # 初始化缩放因子
                self.gpu_ids                   # GPU设备列表
            )

        if self.isTrain:
            # 定义损失函数
            # GAN损失: 对抗训练的核心损失
            self.criterionGAN = network.GANLoss(opt.gan_mode).to(self.device)
            # L1损失: 像素级监督，保持图像细节
            self.criterionL1 = torch.nn.L1Loss()
            # L2损失: 均方误差，用于生成器的监督学习
            self.criterionL2 = torch.nn.MSELoss()
            
            # 定义优化器
            # Adam优化器，学习率和动量参数可配置
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), 
                lr=opt.lr,              # 学习率
                betas=(opt.beta1, 0.999) # 动量参数
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), 
                lr=opt.lr, 
                betas=(opt.beta1, 0.999)
            )
            
            # 将优化器添加到列表中，供BaseModel的学习率调度器使用
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, data, target):
        self.real_A = data.to(self.device)      # 输入图像 A
        self.real_B = target.to(self.device)    # 目标图像 B

    def forward(self):
        self.fake_B = self.netG(self.real_A)  # G(A) 生成假图像
        
        # 可选: 计算网络复杂度 (已注释)
        # self.total_ops, self.total_params = profile(self.netG.cuda(), (self.real_A.cuda(),))
        # print(self.total_ops, self.total_params)

    def backward_D(self):
        # ===== 处理生成图像 (Fake) =====
        # 拼接输入图像和生成图像 (条件GAN的标准做法)
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # 判别器预测 (使用detach()阻止梯度回传到生成器)
        pred_fake = self.netD(fake_AB.detach())
        # 计算假图像的损失 (希望判别器输出False)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # ===== 处理真实图像 (Real) =====
        # 拼接输入图像和真实目标图像
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        # 判别器预测
        pred_real = self.netD(real_AB)
        # 计算真实图像的损失 (希望判别器输出True)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        
        # ===== 总损失和反向传播 =====
        # 组合损失: 平均真假损失
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        # 反向传播计算梯度
        self.loss_D.backward()

    def backward_G(self):
        # ===== 1. GAN对抗损失 =====
        # 生成器希望判别器认为生成图像是真实的
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        # 计算对抗损失 (希望判别器输出True)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # ===== 2. 像素级L2损失 =====
        # 生成图像应该与真实图像在像素级别相似
        # 使用L2损失而非L1损失，L2对大误差惩罚更重
        self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * self.opt.lambda_L1
        
        # ===== 3. 感知损失 (VGG特征损失) =====
        # 使用预训练VGG网络提取高级特征
        # 在特征空间而非像素空间比较图像相似性
        
        # # 提取真实图像的VGG特征
        # self.real_Bf = self.feature_extractor(self.real_B.cpu()).cuda() # [B, 256, H', W']，
        # # 提取生成图像的VGG特征
        # self.fake_Bf = self.feature_extractor(self.fake_B.cpu()).cuda() # [B, 256, H', W']，
        
        # # 拼接特征用于损失计算
        # pred_fake1 = torch.cat((self.real_Bf, self.fake_Bf), 1) # [B, 512, H', W']
        # # 计算感知损失 (特征应该相似)
        # self.contentLoss = self.criterionGAN(pred_fake1, True)

        # ===== 4. 总损失和反向传播 =====
        # 组合所有损失项
        # self.loss_G = self.loss_G_GAN + self.loss_G_L2 + self.contentLoss
        self.loss_G = self.loss_G_GAN + self.loss_G_L2
        # 反向传播计算梯度
        self.loss_G.backward()

    def optimize_parameters(self):
        # ===== 1. 前向传播 =====
        self.forward()  # 计算假图像 G(A)
        
        # ===== 2. 更新判别器 D =====
        # 启用判别器的梯度计算
        self.set_requires_grad(self.netD, True)
        # 清零判别器梯度
        self.optimizer_D.zero_grad()
        # 计算判别器梯度
        self.backward_D()
        # 更新判别器权重
        self.optimizer_D.step()

        # ===== 3. 更新生成器 G =====
        # 禁用判别器梯度计算 (优化生成器时不需要)
        self.set_requires_grad(self.netD, False)
        # 清零生成器梯度
        self.optimizer_G.zero_grad()
        # 计算生成器梯度
        self.backward_G()
        # 更新生成器权重
        self.optimizer_G.step()
