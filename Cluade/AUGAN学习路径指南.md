# AUGAN项目学习路径指南

## 🎯 学习目标
理解AUGAN如何将低质量单角度超声图像转换为高质量多角度超声图像

## 📚 建议学习顺序

### 🥇 第一步：理解项目整体架构 (先看文档)
**从这里开始：**
```bash
# 先读项目概览
cat /home/liujia/dev/AUGAN_725/CLAUDE.md
```

**核心概念：**
- AUGAN = 超声图像增强的条件GAN
- 输入：单角度低质量超声图像  
- 输出：多角度高质量超声图像

### 📊 第二步：理解数据处理 (推荐起点)
**从这个文件开始学习：**
```bash
# 最简单的入门文件 (只有100多行)
code /home/liujia/dev/AUGAN_725/cubdl/example_picmus_torch.py
```

**学习顺序：**

1️⃣ **先看数据格式** (理解超声成像基础)
```bash
# 理解PICMUS数据格式和超声成像原理
code /home/liujia/dev/AUGAN_725/cubdl/README.md
code /home/liujia/dev/AUGAN_725/cubdl/PlaneWaveData.py
```

2️⃣ **然后看DAS算法** (核心重建算法)
```bash
# 理解超声图像重建过程
code /home/liujia/dev/AUGAN_725/cubdl/example_picmus_torch.py  # 从第36行的create_network开始
code /home/liujia/dev/AUGAN_725/cubdl/das_torch.py
```

3️⃣ **接着看数据处理** (训练数据准备)
```bash
# 理解数据预处理流程
code /home/liujia/dev/AUGAN_725/data_process.py  # 从第345行的load_dataset开始
```

### 🧠 第三步：理解模型架构

**学习顺序：**

1️⃣ **模型基类** (理解设计模式)
```bash
code /home/liujia/dev/AUGAN_725/models/base_model.py  # 重点：第16-38行的类设计思想
```

2️⃣ **Pix2Pix模型** (核心GAN实现)
```bash  
code /home/liujia/dev/AUGAN_725/models/pix2pix_model.py  # 重点：第88行__init__和第314行optimize_parameters
```

3️⃣ **网络架构** (U-Net + PatchGAN)
```bash
code /home/liujia/dev/AUGAN_725/models/network.py  # 重点：U-Net生成器和PatchGAN判别器
```

### 🚀 第四步：理解训练流程
**主训练脚本：**
```bash
code /home/liujia/dev/AUGAN_725/train.py  # 重点：第112行开始的训练循环
```

### 🧪 第五步：理解测试流程  
**测试脚本：**
```bash
code /home/liujia/dev/AUGAN_725/test.py  # 重点：第122行开始的推理流程
```

## 🔍 关键概念理解

### 数据流程
```
PICMUS原始数据 → DAS重建 → 单角度图像(输入A) 
                        → 多角度图像(目标B)
                        ↓
                    数据预处理(padding/normalize)
                        ↓  
                    训练数据集(A→B的映射对)
```

### 模型架构
```
输入A → U-Net生成器 → 生成B'
       ↓
[A,B']和[A,B] → PatchGAN判别器 → 真假判别
```

### 损失函数
```
总损失 = GAN损失(对抗) + L2损失(像素) + VGG损失(感知)
```

## 💡 每个文件的学习重点

### 数据理解阶段
- **`cubdl/README.md`** - 超声成像基础概念，PICMUS数据格式
- **`cubdl/example_picmus_torch.py`** - 数据加载(`load_datasets`)、网络创建(`create_network`)、图像重建(`mk_img`)
- **`data_process.py`** - DAS重建(`create_das_reconstructions`)、预处理(`preprocess_images`)、数据集加载(`load_dataset`)

### 模型架构阶段  
- **`models/base_model.py`** - 抽象基类设计、模型生命周期管理、GPU/CPU设备管理
- **`models/pix2pix_model.py`** - 模型初始化、前向传播(`forward`)、判别器训练(`backward_D`)、生成器训练(`backward_G`)、参数优化(`optimize_parameters`)
- **`models/network.py`** - U-Net生成器结构、PatchGAN判别器结构、网络初始化方法

### 训练流程阶段
- **`train.py`** - 数据加载配置、训练循环、损失监控和保存
- **`options/`** - 参数定义和默认值、训练和测试选项区别

### 测试评估阶段
- **`test.py`** - 模型加载和推理、结果保存和可视化
- **`cubdl/metrics.py`** - CNR, gCNR, SNR, PSNR等评估指标

## 💻 实践建议

### 快速上手技巧
**1. 先运行再理解**
```bash
# 用最简单的命令跑一遍，看看输出
python train.py --name learning_test --n_epochs 1 --print_freq 1
```

**2. 关注关键函数**
- `mk_img()` - 图像生成
- `forward()` - 模型前向传播  
- `optimize_parameters()` - 训练一步

**3. 忽略复杂细节**
第一遍学习时可以跳过：
- GPU优化代码
- 复杂的网络初始化  
- 详细的损失计算

### 学习方法建议

#### 第一天：理解数据流程
1. **读代码注释** - 先看函数注释理解作用
2. **画数据流程图** - 手绘数据从输入到输出的路径
3. **运行小例子** - 单独测试每个函数

#### 第二天：理解模型架构
```bash
# 先看这个简单的可视化
python -c "
from models.pix2pix_model import Pix2PixModel
print('Pix2Pix = U-Net生成器 + PatchGAN判别器')
print('输入：单角度图像 → 输出：多角度图像')
"
```

## 🛠️ 调试技巧

- 使用`--print_freq 1`查看每个batch的训练信息
- 检查`./images/experiment_name/`目录的可视化结果
- 监控GPU显存使用情况避免OOM
- 从小数据集和少epoch开始测试

## ❓ 常见问题

1. **数据维度错误**: 检查图像padding是否正确
2. **GPU内存不足**: 减小batch_size或使用CPU
3. **训练不收敛**: 调整学习率和损失权重
4. **生成图像模糊**: 检查感知损失是否生效

## 🏁 学习检查点

### 第一阶段检查 - 数据理解
- [ ] 理解PICMUS数据格式(idata, qdata, angles等)
- [ ] 明白DAS重建算法的作用
- [ ] 知道单角度vs多角度图像的区别
- [ ] 理解数据预处理流程(padding, normalize)

### 第二阶段检查 - 模型架构  
- [ ] 理解Pix2Pix的基本原理
- [ ] 知道U-Net生成器的结构特点
- [ ] 明白PatchGAN判别器的作用
- [ ] 理解条件GAN与普通GAN的区别

### 第三阶段检查 - 训练流程
- [ ] 理解GAN的对抗训练过程
- [ ] 知道损失函数的三个组成部分
- [ ] 明白为什么要交替优化G和D
- [ ] 理解学习率调度和模型保存

### 第四阶段检查 - 测试评估
- [ ] 知道如何加载训练好的模型
- [ ] 理解推理过程和结果保存
- [ ] 明白各种评估指标的含义
- [ ] 会分析训练结果的好坏

## 💡 学习心态

- 🎯 **目标明确**：理解"单角度→多角度"的图像转换
- 🏃 **先跑后懂**：先让代码跑起来，再深入理解细节  
- 📝 **多做笔记**：画流程图，记录关键概念
- 🤝 **随时提问**：遇到不懂的地方就问

**记住：学习代码最好的方法就是运行它、修改它、理解它！** 🚀

现在可以从 `cubdl/example_picmus_torch.py` 开始你的AUGAN学习之旅！