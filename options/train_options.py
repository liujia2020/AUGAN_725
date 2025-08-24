
#!/usr/bin/env python3
"""
AUGAN 训练配置选项模块 - 详细注释版
train_options.py - 定义训练专用的配置参数
继承BaseOptions，添加训练相关的优化器、学习率、保存策略等设置
"""

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """
    训练选项类 - 包含训练专用配置

    同时包含BaseOptions中定义的共享选项。
    """

    def initialize(self, parser):
        """
        添加训练专用的选项参数
        """
        parser = BaseOptions.initialize(self, parser)

        # ==================== 训练显示设置 ====================
        parser.add_argument('--print_freq', type=int, default=20,
                        help='在控制台显示训练结果的频率')

        # ==================== 数据集加载设置 ====================
        # 注释掉的参数供参考
        # parser.add_argument('-b',"--batch_size",type=int,default=10, help="每个训练epoch的批次大小")
        # parser.add_argument('-n',"--num_epoch",type=int,default=100,help="训练epoch数量")
        # parser.add_argument('-l',"--learning_rate",type=float,default=0.0001,help="学习率")

        parser.add_argument('-f', '--load', type=str, default='./img_data',
                        help='从文件加载模型')

        parser.add_argument('-s','--scale',type=float,default=0.5,
                        help='图像的下采样因子')

        parser.add_argument('-v', '--validation',type=float, default=10.0,
                        help='用作验证的数据百分比 (0-100)')

        # ==================== 网络保存和加载参数 ====================
        parser.add_argument('--save_latest_freq', type=int, default=5000,
                        help='保存最新结果的频率')

        parser.add_argument('--save_epoch_freq', type=int, default=1,
                        help='在epoch结束时保存checkpoints的频率')

        parser.add_argument('--save_by_iter', action='store_true',
                        help='是否按迭代保存模型')

        parser.add_argument('--continue_train', action='store_true',
                        help='继续训练: 加载最新模型')

        parser.add_argument('--epoch_count', type=int, default=1,
                        help='起始epoch计数，我们按<epoch_count>,<epoch_count>+<save_latest_freq>,...保存模型')

        parser.add_argument('--phase', type=str, default='train',
                        help='train, val, test, 等')

        # ==================== 训练参数 ====================
        parser.add_argument('--niter', type=int, default=1,
                        help='在起始学习率下的迭代次数')

        parser.add_argument('--n_epochs', type=int, default=1,
                        help='使用初始学习率的epoch数量')

        parser.add_argument('--niter_decay', type=int, default=1,
                        help='线性衰减学习率到零的迭代次数')

        parser.add_argument('--beta1', type=float, default=0.5,
                        help='adam的动量项')

        parser.add_argument('--lr', type=float, default=0.0002,
                        help='adam的初始学习率')

        parser.add_argument('--gan_mode', type=str, default='vanilla',
                        help='GAN目标的类型. [vanilla| lsgan | wgangp]. vanilla GAN损失是原始GAN论文中使用的交叉熵目标。')

        parser.add_argument('--pool_size', type=int, default=50,
                        help='存储之前生成图像的图像缓冲区大小')

        parser.add_argument('--lr_policy', type=str, default='linear',
                        help='学习率策略. [linear | step | plateau | cosine]')

        parser.add_argument('--lr_decay_iters', type=int, default=50,
                        help='每lr_decay_iters次迭代乘以gamma')

        self.isTrain = True  # 训练模式
        return parser