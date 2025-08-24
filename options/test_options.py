# from .base_options import BaseOptions


# class TestOptions(BaseOptions):
#     """This class includes test options.

#     It also includes shared options defined in BaseOptions.
#     """

#     def initialize(self, parser):
#         parser = BaseOptions.initialize(self, parser)  # define shared options
#         parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
#         parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
#         parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
#         parser.add_argument('-f', '--load', type=str, default='./img_data',
#                             help='Load model from a file')
#         # Dropout and Batchnorm has different behavioir during training and test.
#         parser.add_argument('--eval', default=True, help='use eval mode during test time.')
#         parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
#         parser.set_defaults(name='unet_p10')
#         parser.set_defaults(epoch='latest')
#         # rewrite devalue values
#         parser.set_defaults(model='test',netG='unet_128')
#         # To avoid cropping, the load_size should be the same as crop_size
#         parser.set_defaults(load_size=parser.get_default('crop_size'))
#         self.isTrain = False
#         return parser

#!/usr/bin/env python3
"""
AUGAN 测试配置选项模块 - 详细注释版
test_options.py - 定义测试/推理专用的配置参数
继承BaseOptions，添加测试相关的结果保存、评估模式等设置
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """
    测试选项类 - 包含测试专用配置

    同时包含BaseOptions中定义的共享选项。
    """

    def initialize(self, parser):
        """
        添加测试专用的选项参数
        """
        parser = BaseOptions.initialize(self, parser)  # 定义共享选项

        # ==================== 测试结果保存设置 ====================
        parser.add_argument('--results_dir', type=str, default='./results/',
                        help='在这里保存结果.')

        parser.add_argument('--aspect_ratio', type=float, default=1.0,
                        help='结果图像的宽高比')

        parser.add_argument('--phase', type=str, default='test',
                        help='train, val, test, 等')

        parser.add_argument('-f', '--load', type=str, default='./img_data',
                        help='从文件加载模型')

        # ==================== 网络评估模式设置 ====================
        # Dropout和Batchnorm在训练和测试期间具有不同的行为。
        parser.add_argument('--eval', default=True,
                        help='在测试时使用eval模式.')

        parser.add_argument('--num_test', type=int, default=50,
                        help='要运行多少测试图像')

        # ==================== 默认值设置 ====================
        parser.set_defaults(name='unet_p10')    # 默认实验名称
        parser.set_defaults(epoch='latest')     # 加载最新模型

        # 重写默认值
        parser.set_defaults(model='test', netG='unet_128')

        # 为了避免裁剪，load_size应该与crop_size相同
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        self.isTrain = False  # 测试模式
        return parser