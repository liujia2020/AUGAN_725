#!/usr/bin/env python3
"""
AUGAN 基础模型类详细注释版本
base_model.py - 所有模型的抽象基类，定义了模型的基本接口和通用功能
提供模型训练、保存、加载等通用方法
"""

import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import network
from thop import profile


class BaseModel(ABC):
    """
    模型基类 (抽象基类)
    
    设计目的:
        为所有模型(Pix2Pix, CycleGAN等)提供统一的接口和通用功能
        封装模型训练、测试、保存、加载等常用操作
        使用抽象方法强制子类实现核心功能
    
    主要功能:
        1. 设备管理 (CPU/GPU)
        2. 模型保存和加载
        3. 学习率调度
        4. 网络参数控制
        5. 损失值监控
        6. 网络信息打印
    
    继承关系:
        BaseModel (抽象基类)
        ├── Pix2PixModel (AUGAN使用)
        ├── CycleGANModel
        └── TestModel
    """
    
    def __init__(self, opt):
        """
        初始化基础模型
        
        参数:
            opt: 训练选项对象，包含所有配置参数
        
        初始化内容:
            1. 基本配置设置
            2. 设备配置 (GPU/CPU)
            3. 路径设置
            4. 空列表初始化
        """
        # 保存配置选项
        self.opt = opt
        
        # GPU设备配置
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        
        # 设备选择: 如果有GPU则使用第一个GPU，否则使用CPU
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        
        # 模型保存目录: checkpoints_dir/experiment_name/
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        
        # CUDNN基准测试优化 (注释掉，因为可能影响某些预处理操作)
        # if opt.preprocess != 'scale_width':
        #     torch.backends.cudnn.benchmark = True
        
        # 初始化空列表，由子类填充
        self.loss_names = []      # 损失函数名称列表，用于打印和保存
        self.model_names = []     # 模型名称列表，用于保存和加载网络
        self.visual_names = []    # 可视化图像名称列表
        self.optimizers = []      # 优化器列表
        self.image_paths = []     # 图像路径列表
        self.metric = 0          # 用于学习率策略 'plateau' 的指标

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        修改命令行选项 (静态方法)
        
        功能:
            允许子类添加模型特定的命令行参数
            重写现有选项的默认值
        
        参数:
            parser: 原始命令行解析器
            is_train (bool): 是否为训练模式
        
        返回:
            修改后的解析器
        
        使用方式:
            子类重写此方法来添加自己的参数
            例如: parser.add_argument('--lambda_L1', type=float, default=100)
        """
        return parser

    @abstractmethod
    def forward(self):
        """
        前向传播 (抽象方法)
        
        要求:
            所有子类必须实现此方法
            定义模型的前向计算过程
        
        调用时机:
            - 训练时的optimize_parameters()中
            - 测试时的test()中
        
        典型实现:
            self.fake_B = self.netG(self.real_A)
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        参数优化 (抽象方法)
        
        要求:
            所有子类必须实现此方法
            定义损失计算、反向传播和参数更新过程
        
        典型实现:
            1. forward()                     # 前向传播
            2. set_requires_grad(netD, True) # 启用判别器梯度
            3. optimizer_D.zero_grad()       # 清零梯度
            4. backward_D()                  # 判别器反向传播
            5. optimizer_D.step()            # 更新判别器
            6. 重复3-5步骤更新生成器
        """
        pass

    def setup(self, opt):
        """
        模型设置
        
        功能:
            1. 创建学习率调度器
            2. 加载预训练模型 (如果需要)
            3. 打印网络信息
        
        参数:
            opt: 训练选项对象
        
        执行顺序:
            1. 如果是训练模式，为每个优化器创建调度器
            2. 如果是测试模式或继续训练，加载预训练模型
            3. 打印网络结构和参数信息
        """
        if self.isTrain:
            # 为每个优化器创建学习率调度器
            self.schedulers = [network.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        
        # 加载预训练模型的条件:
        # 1. 测试模式 (不训练，只推理)
        # 2. 继续训练 (从checkpoint恢复)
        if not self.isTrain or opt.continue_train:
            # 确定加载的模型版本
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        
        # 打印网络信息 (可选详细模式)
        self.print_networks(opt.verbose)

    def load_networks(self, epoch):
        """
        从磁盘加载所有网络
        
        参数:
            epoch (int/str): 模型版本标识，用于文件名 'epoch_net_name.pth'
        
        加载过程:
            1. 遍历所有模型名称
            2. 构造文件路径
            3. 加载state_dict
            4. 处理兼容性问题
            5. 应用到网络
        
        文件命名规则:
            格式: {epoch}_net_{name}.pth
            例如: 100_net_G.pth, 100_net_D.pth
        """
        for name in self.model_names:
            if isinstance(name, str):
                # 构造加载文件名
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                
                # 获取对应的网络对象
                net = getattr(self, 'net' + name)  # 例如: self.netG, self.netD
                
                # 处理DataParallel包装的网络
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                
                print('loading the model from %s' % load_path)
                
                # 加载state_dict，指定设备映射
                state_dict = torch.load(load_path, map_location=str(self.device))
                
                # 删除metadata (如果存在)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # 修复InstanceNorm的兼容性问题 (PyTorch < 0.4)
                for key in list(state_dict.keys()):
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                
                # 将state_dict应用到网络
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """
        打印网络信息
        
        参数:
            verbose (bool): 是否打印详细的网络架构
        
        打印内容:
            1. 网络名称和参数数量
            2. 如果verbose=True，打印完整网络结构
        
        参数计算:
            统计所有可训练参数的数量
            以百万(M)为单位显示
        """
        print('---------- Networks initialized -------------')
        
        for name in self.model_names:
            if isinstance(name, str):
                # 获取网络对象
                net = getattr(self, 'net' + name)
                
                # 计算参数数量
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()  # 参数元素数量
                
                # 如果verbose=True，打印网络结构
                if verbose:
                    print(net)
                
                # 打印参数数量 (以百万为单位)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        
        print('-----------------------------------------------')

    def update_learning_rate(self):
        """
        更新学习率
        
        调用时机:
            每个epoch结束时调用
        
        更新过程:
            1. 记录当前学习率
            2. 调用所有调度器的step()方法
            3. 获取新学习率
            4. 打印学习率变化
        
        特殊处理:
            plateau策略需要传入监控指标
        """
        # 记录更新前的学习率
        old_lr = self.optimizers[0].param_groups[0]['lr']
        
        # 更新所有调度器
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                # ReduceLROnPlateau需要传入监控指标
                scheduler.step(self.metric)
            else:
                # 其他调度器直接step
                scheduler.step()

        # 获取更新后的学习率
        lr = self.optimizers[0].param_groups[0]['lr']
        
        # 打印学习率变化
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def set_requires_grad(self, nets, requires_grad=False):
        """
        设置网络的梯度计算开关
        
        功能:
            控制网络参数是否需要计算梯度
            用于优化内存使用和计算效率
        
        参数:
            nets: 网络或网络列表
            requires_grad (bool): 是否需要计算梯度
        
        使用场景:
            1. 优化生成器时，禁用判别器梯度: set_requires_grad(netD, False)
            2. 优化判别器时，启用判别器梯度: set_requires_grad(netD, True)
        
        内存优化:
            禁用不需要的梯度计算可以节省大量显存
        """
        # 确保nets是列表格式
        if not isinstance(nets, list):
            nets = [nets]
        
        # 设置每个网络的所有参数
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_losses(self):
        """
        获取当前损失值
        
        功能:
            收集所有损失项的当前值
            用于训练日志和可视化
        
        返回:
            OrderedDict: 损失名称 -> 损失值的有序字典
        
        损失名称:
            由子类的loss_names列表定义
            例如: ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        
        使用方式:
            losses = model.get_current_losses()
            print(f"G_GAN: {losses['G_GAN']:.4f}")
        """
        errors_ret = OrderedDict()
        
        for name in self.loss_names:
            if isinstance(name, str):
                # 获取对应的损失属性值
                # 例如: loss_G_GAN -> self.loss_G_GAN
                loss_value = getattr(self, 'loss_' + name)
                errors_ret[name] = float(loss_value)  # 转换为Python float
        
        return errors_ret

    def save_networks(self, epoch):
        """
        保存所有网络到磁盘
        
        参数:
            epoch (int): 当前epoch，用于文件命名
        
        保存过程:
            1. 遍历所有模型
            2. 构造保存路径
            3. 处理GPU/CPU兼容性
            4. 保存state_dict
        
        文件命名:
            格式: {epoch}_net_{name}.pth
            例如: 100_net_G.pth
        
        设备处理:
            保存前将模型移到CPU，保存后恢复到GPU
            确保保存的模型可以在不同设备上加载
        """
        for name in self.model_names:
            if isinstance(name, str):
                # 构造保存文件名
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                
                # 获取网络对象
                net = getattr(self, 'net' + name)

                # 处理GPU/CPU保存
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # GPU模式: 保存CPU版本，然后恢复到GPU
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])  # 恢复到GPU
                else:
                    # CPU模式: 直接保存
                    torch.save(net.cpu().state_dict(), save_path)

    def eval(self):
        """
        设置所有模型为评估模式
        
        功能:
            调用所有网络的eval()方法
            禁用Dropout和BatchNorm的训练模式行为
        
        调用时机:
            测试/验证时调用
        
        影响:
            - Dropout层不再随机丢弃神经元
            - BatchNorm使用累积的统计信息而非当前batch
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """
        测试模式的前向传播
        
        功能:
            在no_grad()上下文中执行forward()
            用于推理时，不保存中间梯度信息
        
        内存优化:
            torch.no_grad()禁用梯度计算，节省显存
        
        使用场景:
            - 模型推理
            - 验证集评估
            - 生成测试结果
        """
        with torch.no_grad():
            self.forward()

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """
        修复InstanceNorm兼容性问题 (私有方法)
        
        背景:
            PyTorch 0.4之前的InstanceNorm保存格式与新版本不兼容
            需要移除某些不存在的属性
        
        参数:
            state_dict: 模型状态字典
            module: 当前模块
            keys: 键路径列表
            i: 当前键索引
        
        修复内容:
            1. 移除InstanceNorm的running_mean和running_var (如果不存在)
            2. 移除num_batches_tracked属性
        
        递归处理:
            深度优先遍历模块树，修复所有InstanceNorm层
        """
        key = keys[i]
        
        if i + 1 == len(keys):  # 到达叶子节点 (参数/缓冲区)
            # 修复InstanceNorm的running_mean和running_var
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            
            # 移除num_batches_tracked属性
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            # 递归处理子模块
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


# ==================== BaseModel总结 ====================
"""
BaseModel.py 核心功能总结:

1. **抽象基类设计**:
   - 使用ABC确保子类实现核心方法
   - 提供统一的模型接口
   - 封装通用功能，减少代码重复

2. **设备管理**:
   - 自动检测GPU/CPU
   - 处理多GPU并行
   - 保存/加载时的设备兼容性

3. **模型生命周期**:
   - 初始化: __init__() + setup()
   - 训练: forward() + optimize_parameters()
   - 保存: save_networks()
   - 加载: load_networks()
   - 测试: eval() + test()

4. **学习率管理**:
   - 支持多种调度策略
   - 自动更新所有优化器
   - 打印学习率变化

5. **内存优化**:
   - 梯度开关控制
   - no_grad()测试模式
   - CPU/GPU状态管理

6. **兼容性处理**:
   - InstanceNorm历史版本兼容
   - PyTorch版本适配
   - DataParallel支持

在AUGAN中的作用:
- Pix2PixModel继承BaseModel
- 提供训练、测试、保存等基础功能
- 简化模型实现，专注于核心逻辑
- 确保代码的可维护性和扩展性

BaseModel是整个框架的基石，所有具体模型都依赖它提供的基础设施。
"""