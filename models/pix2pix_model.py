#!/usr/bin/env python3
"""
AUGAN Pix2Pix模型详细注释版本
pix2pix_model.py - AUGAN的核心训练模型，实现条件GAN的完整训练逻辑
基于Pix2Pix框架，专门针对超声图像增强任务进行了优化
"""

import torch
from .base_model import BaseModel
from . import network
import torchvision
from thop import profile


class Pix2PixModel(BaseModel):
    """
    Pix2Pix模型 - AUGAN的核心训练模型
    
    模型原理:
        Pix2Pix是条件生成对抗网络(cGAN)的经典实现
        学习从输入图像到输出图像的映射关系
        使用配对数据进行监督学习
    
    网络架构:
        - 生成器: U-Net (默认unet_256，8层下采样)
        - 判别器: PatchGAN (默认basic，70×70感受野)
        - 损失函数: GAN损失 + L1损失 + 感知损失
    
    在AUGAN中的应用:
        - 输入A: 单角度超声重建图像 (低质量)
        - 输出B: 多角度复合超声图像 (高质量)
        - 目标: 学习从低质量到高质量的超声图像映射
    
    训练目标:
        Loss = λ_GAN * L_GAN + λ_L1 * L_L1 + λ_content * L_content
        - L_GAN: 对抗损失，使生成图像欺骗判别器
        - L_L1: 像素级L1损失，保持图像细节
        - L_content: 感知损失，保持高级语义特征
    
    优势:
        - 条件生成，有监督学习，训练稳定
        - U-Net保持细节，PatchGAN关注纹理
        - 多重损失约束，生成质量高
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        修改命令行选项，设置Pix2Pix模型的默认参数
        
        参数:
            parser: 命令行解析器
            is_train (bool): 是否为训练模式
        
        返回:
            修改后的解析器
        
        Pix2Pix默认配置:
            - 规范化: instance (适合图像翻译)
            - 生成器: unet_128 (AUGAN实际使用unet_256)
            - 判别器: basic PatchGAN
            - 注意力: 不使用SAB
            - 池化: 不使用图像池
            - GAN模式: vanilla (原始GAN损失)
            - L1权重: 1.0 (L1损失的重要性)
        """
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
        """
        初始化Pix2Pix模型
        
        参数:
            opt: 配置选项对象，包含所有实验参数
        
        初始化内容:
            1. 继承BaseModel的基础功能
            2. 定义损失名称 (用于监控训练)
            3. 创建特征提取器 (用于感知损失)
            4. 创建生成器和判别器网络
            5. 定义损失函数和优化器
        """
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
            opt.ngf,            # 生成器滤波器数量 (通常64)
            opt.netG,           # 生成器架构 (unet_256)
            opt.norm,           # 规范化类型 (instance)
            not opt.no_dropout, # 是否使用dropout
            opt.init_type,      # 权重初始化方法
            opt.init_gain,      # 初始化缩放因子
            self.gpu_ids,       # GPU设备列表
            opt.use_sab         # 是否使用自注意力
        )

        if self.isTrain:
            # 训练模式: 定义判别器
            # 条件GAN需要同时输入原图和目标图，所以通道数是两者之和
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
        """
        设置模型输入数据
        
        参数:
            data: 输入数据 (低分辨率/低质量图像)
            target: 目标数据 (高分辨率/高质量图像)
        
        功能:
            1. 将数据移动到指定设备 (GPU/CPU)
            2. 设置为模型的输入属性
        
        在AUGAN中:
            - data: 单角度超声重建图像 (低质量)
            - target: 多角度复合超声图像 (高质量，ground truth)
        """
        self.real_A = data.to(self.device)      # 输入图像 A
        self.real_B = target.to(self.device)    # 目标图像 B

    def forward(self):
        """
        前向传播
        
        功能:
            通过生成器网络生成假图像
            G(A) → fake_B
        
        调用时机:
            - 训练时的optimize_parameters()中
            - 测试时的test()中
        
        计算流程:
            real_A (输入) → netG → fake_B (生成)
        
        注意:
            注释掉的profile代码用于计算网络的FLOPs和参数量
            可用于分析模型复杂度
        """
        self.fake_B = self.netG(self.real_A)  # G(A) 生成假图像
        
        # 可选: 计算网络复杂度 (已注释)
        # self.total_ops, self.total_params = profile(self.netG.cuda(), (self.real_A.cuda(),))
        # print(self.total_ops, self.total_params)

    def backward_D(self):
        """
        判别器反向传播
        
        功能:
            计算判别器的GAN损失并进行反向传播
            训练判别器区分真实图像和生成图像
        
        损失计算:
            L_D = 0.5 * [L_D_fake + L_D_real]
            - L_D_fake: 判别器对生成图像的损失 (希望输出False)
            - L_D_real: 判别器对真实图像的损失 (希望输出True)
        
        关键技术点:
            1. 使用detach()阻止梯度回传到生成器
            2. 条件GAN: 同时输入原图和目标图
            3. 损失加权0.5: 标准GAN训练技巧
        """
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
        """
        生成器反向传播
        
        功能:
            计算生成器的总损失并进行反向传播
            训练生成器生成高质量的假图像
        
        损失组成:
            L_G = L_G_GAN + λ_L1 * L_G_L2 + λ_content * L_content
            1. L_G_GAN: 对抗损失，欺骗判别器
            2. L_G_L2: 像素级L2损失，保持图像相似性
            3. L_content: 感知损失，保持高级特征一致性
        
        创新点:
            AUGAN使用VGG特征提取器计算感知损失
            提升生成图像的视觉质量和语义一致性
        """
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
        
        # 提取真实图像的VGG特征
        # 需要先移动到CPU再移动回GPU (可能是为了兼容性)
        self.real_Bf = self.feature_extractor(self.real_B.cpu()).cuda()
        # 提取生成图像的VGG特征
        self.fake_Bf = self.feature_extractor(self.fake_B.cpu()).cuda()
        
        # 拼接特征用于损失计算
        pred_fake1 = torch.cat((self.real_Bf, self.fake_Bf), 1)
        # 计算感知损失 (特征应该相似)
        self.contentLoss = self.criterionGAN(pred_fake1, True)

        # ===== 4. 总损失和反向传播 =====
        # 组合所有损失项
        self.loss_G = self.loss_G_GAN + self.loss_G_L2 + self.contentLoss
        # 反向传播计算梯度
        self.loss_G.backward()

    def optimize_parameters(self):
        """
        优化网络参数 - 完整的训练步骤
        
        功能:
            执行一个完整的训练迭代
            交替优化判别器和生成器
        
        训练流程:
            1. 前向传播: 生成假图像
            2. 优化判别器: 学习区分真假
            3. 优化生成器: 学习生成更好的假图像
        
        关键技术:
            - 使用set_requires_grad控制梯度计算
            - 先更新判别器，再更新生成器
            - 每次都清零梯度，避免累积
        
        这是GAN训练的标准流程，确保对抗训练的稳定性
        """
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


# ==================== Pix2Pix模型总结 ====================
"""
pix2pix_model.py 核心功能总结:

1. **模型架构**:
   - 继承BaseModel，获得基础模型功能
   - 生成器: U-Net (保持细节的编码-解码结构)
   - 判别器: PatchGAN (局部纹理判别)

2. **损失函数**:
   - GAN损失: 对抗训练的核心
   - L2损失: 像素级监督 (AUGAN使用L2而非L1)
   - 感知损失: VGG特征损失 (AUGAN创新)

3. **训练策略**:
   - 条件GAN: 使用配对数据监督学习
   - 交替优化: 先D后G的标准GAN训练
   - 梯度控制: 精确控制梯度流向

4. **AUGAN特点**:
   - 单通道输入输出 (灰度超声图像)
   - VGG感知损失 (提升视觉质量)
   - 实例规范化 (适合图像翻译)

5. **数据流**:
   输入A (单角度) → 生成器 → 输出B' (增强)
   [A, B'] vs [A, B] → 判别器 → 真假判别

这个模型实现了从低质量单角度超声图像到高质量
多角度复合超声图像的端到端学习。
"""