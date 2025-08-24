#!/usr/bin/env python3
"""
AUGAN学习练习 - 第三步：理解GAN的基本原理
目标：通过简单例子理解生成对抗网络，为学习AUGAN的Pix2Pix架构做准备
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print("🎯 AUGAN学习练习 - 第三步：GAN基本原理")
print("="*50)

# ===== 第一部分：理解GAN的基本概念 =====
print("🧠 第一部分：GAN的基本思想")
print("   GAN = Generative Adversarial Network (生成对抗网络)")
print("   核心思想：两个网络相互对抗，共同进步")
print("   - 生成器(Generator): 试图生成逼真的假数据")
print("   - 判别器(Discriminator): 试图区分真假数据")
print("   - 对抗过程：生成器想骗过判别器，判别器想不被骗")

# ===== 第二部分：创建简单的生成器和判别器 =====
print("\n🏗️  第二部分：构建简单的生成器和判别器")

class SimpleGenerator(nn.Module):
    """
    简单的生成器：将噪声转换成图像
    在AUGAN中：将单角度图像转换成多角度图像
    """
    def __init__(self, input_size=100, output_size=64*64):
        super(SimpleGenerator, self).__init__()
        self.network = nn.Sequential(
            # 噪声 → 隐藏层1
            nn.Linear(input_size, 256),
            nn.ReLU(),
            # 隐藏层1 → 隐藏层2
            nn.Linear(256, 512),
            nn.ReLU(),
            # 隐藏层2 → 输出图像
            nn.Linear(512, output_size),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
    
    def forward(self, noise):
        """前向传播：噪声 → 生成图像"""
        return self.network(noise)

class SimpleDiscriminator(nn.Module):
    """
    简单的判别器：判断图像是真是假
    在AUGAN中：判断[输入图像,输出图像]对是否匹配
    """
    def __init__(self, input_size=64*64):
        super(SimpleDiscriminator, self).__init__()
        self.network = nn.Sequential(
            # 图像 → 隐藏层1
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            # 隐藏层1 → 隐藏层2
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            # 隐藏层2 → 真假判断
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出概率 [0, 1]
        )
    
    def forward(self, image):
        """前向传播：图像 → 真假概率"""
        return self.network(image)

# 创建网络实例
print("   创建生成器和判别器...")
generator = SimpleGenerator(input_size=100, output_size=64*64)
discriminator = SimpleDiscriminator(input_size=64*64)

print(f"   生成器参数数量: {sum(p.numel() for p in generator.parameters()):,}")
print(f"   判别器参数数量: {sum(p.numel() for p in discriminator.parameters()):,}")

# ===== 第三部分：模拟GAN的对抗训练过程 =====
print("\n⚔️  第三部分：模拟GAN对抗训练")

# 创建优化器
lr = 0.0002
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 损失函数
criterion = nn.BCELoss()  # 二元交叉熵损失

# 创建一些"真实"数据 (模拟多角度高质量图像)
def create_real_data(batch_size=32):
    """创建模拟的真实图像数据"""
    # 模拟一些有结构的图像 (比随机噪声更像真实图像)
    real_data = []
    for i in range(batch_size):
        # 创建简单的几何图案
        img = np.zeros((64, 64))
        # 添加圆形
        center_x, center_y = np.random.randint(20, 44, 2)
        radius = np.random.randint(5, 15)
        y, x = np.ogrid[:64, :64]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        img[mask] = 1.0
        # 添加噪声
        img += np.random.normal(0, 0.1, (64, 64))
        real_data.append(img.flatten())
    
    return torch.FloatTensor(real_data)

# 训练标签
real_label = 1.0  # 真实图像的标签
fake_label = 0.0  # 生成图像的标签

print("   开始对抗训练演示...")
batch_size = 16
num_epochs = 50

# 记录训练过程
g_losses = []
d_losses = []
generated_samples = []

for epoch in range(num_epochs):
    # ===== 训练判别器 =====
    # 1. 训练判别器识别真实数据
    discriminator.zero_grad()
    
    # 真实数据
    real_data = create_real_data(batch_size)
    real_labels = torch.full((batch_size, 1), real_label)
    
    # 判别器对真实数据的判断
    real_output = discriminator(real_data)
    d_loss_real = criterion(real_output, real_labels)
    d_loss_real.backward()
    
    # 2. 训练判别器识别生成数据
    # 生成假数据
    noise = torch.randn(batch_size, 100)
    fake_data = generator(noise)
    fake_labels = torch.full((batch_size, 1), fake_label)
    
    # 判别器对生成数据的判断 (注意detach，不更新生成器)
    fake_output = discriminator(fake_data.detach())
    d_loss_fake = criterion(fake_output, fake_labels)
    d_loss_fake.backward()
    
    # 更新判别器
    optimizer_D.step()
    d_loss = d_loss_real + d_loss_fake
    
    # ===== 训练生成器 =====
    generator.zero_grad()
    
    # 生成器希望判别器认为它生成的是真实数据
    fake_output = discriminator(fake_data)
    real_labels_for_g = torch.full((batch_size, 1), real_label)  # 生成器希望输出接近1
    g_loss = criterion(fake_output, real_labels_for_g)
    g_loss.backward()
    
    # 更新生成器
    optimizer_G.step()
    
    # 记录损失
    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())
    
    # 每10个epoch保存一个生成样本
    if epoch % 10 == 0:
        with torch.no_grad():
            sample_noise = torch.randn(1, 100)
            sample_generated = generator(sample_noise)
            generated_samples.append(sample_generated.numpy().reshape(64, 64))
        
        print(f"   Epoch {epoch:2d}: G_loss={g_loss.item():.4f}, D_loss={d_loss.item():.4f}")

print("   对抗训练演示完成！")

# ===== 第四部分：可视化训练过程和结果 =====
print("\n📈 第四部分：可视化GAN训练过程")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 第一行：显示生成器的进化过程
for i, sample in enumerate(generated_samples):
    if i < 4:
        axes[0, i].imshow(sample, cmap='gray')
        axes[0, i].set_title(f'Epoch {i*10}\n生成器输出')
        axes[0, i].axis('off')

# 第二行左：损失函数变化
axes[1, 0].plot(g_losses, 'b-', label='生成器损失', linewidth=2)
axes[1, 0].plot(d_losses, 'r-', label='判别器损失', linewidth=2)
axes[1, 0].set_title('GAN训练过程中的损失变化')
axes[1, 0].set_xlabel('训练步数')
axes[1, 0].set_ylabel('损失值')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 第二行中：真实数据样本
real_sample = create_real_data(1).numpy().reshape(64, 64)
axes[1, 1].imshow(real_sample, cmap='gray')
axes[1, 1].set_title('真实数据样本\n(目标质量)')
axes[1, 1].axis('off')

# 第二行右：最终生成样本
if generated_samples:
    axes[1, 2].imshow(generated_samples[-1], cmap='gray')
    axes[1, 2].set_title('最终生成样本\n(学到的质量)')
    axes[1, 2].axis('off')

# 第二行最右：对抗过程示意图
axes[1, 3].text(0.1, 0.8, '🎯 GAN对抗过程:', fontsize=12, weight='bold')
axes[1, 3].text(0.1, 0.7, '1. 生成器生成假数据', fontsize=10)
axes[1, 3].text(0.1, 0.6, '2. 判别器判断真假', fontsize=10)
axes[1, 3].text(0.1, 0.5, '3. 根据结果更新网络', fontsize=10)
axes[1, 3].text(0.1, 0.4, '4. 重复直到平衡', fontsize=10)
axes[1, 3].text(0.1, 0.2, '📊 最终效果:', fontsize=12, weight='bold')
axes[1, 3].text(0.1, 0.1, '生成器能生成逼真数据', fontsize=10)
axes[1, 3].set_xlim(0, 1)
axes[1, 3].set_ylim(0, 1)
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('/home/liujia/dev/AUGAN_725/Cluade/practice/step3_GAN训练过程.png', dpi=150, bbox_inches='tight')
print("   GAN训练过程图已保存到: step3_GAN训练过程.png")
plt.show()

# ===== 第五部分：从普通GAN到条件GAN (Pix2Pix) =====
print("\n🔀 第五部分：从GAN到条件GAN的进化")

print("   普通GAN的局限:")
print("   - 输入：随机噪声")
print("   - 输出：随机生成的图像")
print("   - 问题：无法控制生成什么样的图像")
print("")
print("   条件GAN (cGAN) 的改进:")
print("   - 输入：条件信息 + 噪声")
print("   - 输出：符合条件的图像")
print("   - 优势：可以控制生成内容")
print("")
print("   Pix2Pix (AUGAN使用的架构):")
print("   - 输入条件：单角度超声图像")
print("   - 输出目标：多角度超声图像")
print("   - 训练数据：成对的 (单角度, 多角度) 图像")

# 模拟条件GAN的概念
class ConditionalGenerator(nn.Module):
    """条件生成器：根据输入条件生成对应输出"""
    def __init__(self, condition_size=64*64, output_size=64*64):
        super(ConditionalGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(condition_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Tanh()
        )
    
    def forward(self, condition):
        """条件输入 → 对应输出"""
        return self.network(condition)

class ConditionalDiscriminator(nn.Module):
    """条件判别器：判断 (输入,输出) 对是否匹配"""
    def __init__(self, condition_size=64*64, image_size=64*64):
        super(ConditionalDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(condition_size + image_size, 512),  # 输入和输出拼接
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, condition, image):
        """(条件, 图像) → 匹配概率"""
        combined = torch.cat([condition, image], dim=1)
        return self.network(combined)

print("\n   条件GAN网络结构:")
cond_generator = ConditionalGenerator()
cond_discriminator = ConditionalDiscriminator()
print(f"   条件生成器参数: {sum(p.numel() for p in cond_generator.parameters()):,}")
print(f"   条件判别器参数: {sum(p.numel() for p in cond_discriminator.parameters()):,}")

# ===== 总结 =====
print("\n" + "="*50)
print("🎓 第三步学习总结:")
print("   1. GAN原理：生成器vs判别器的对抗训练")
print("   2. 训练过程：两个网络相互博弈，共同提高")
print("   3. 普通GAN：噪声 → 随机图像")
print("   4. 条件GAN：条件 → 对应图像")
print("   5. Pix2Pix：单角度图像 → 多角度图像")
print("\n💡 现在你理解了AUGAN使用的技术基础:")
print("   - 基础：GAN的对抗训练思想")
print("   - 进阶：条件GAN的有监督学习")
print("   - 应用：Pix2Pix的图像到图像翻译")
print("\n✅ 第三步完成！接下来运行 step4_Pix2Pix架构练习.py")
print("="*50)