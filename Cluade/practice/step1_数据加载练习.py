#!/usr/bin/env python3
"""
AUGAN学习练习 - 第一步：理解超声数据结构
目标：理解什么是超声数据，什么是角度、传感器、采样点
"""

import numpy as np
import matplotlib.pyplot as plt

print("🎯 AUGAN学习练习 - 第一步：数据结构理解")
print("="*50)

# ===== 第一部分：理解超声数据的基本结构 =====
print("📊 第一部分：超声数据的三个维度")

# 超声成像的基本参数
angles = 75      # 发射角度数：从-37°到+37°，共75个角度
elements = 128   # 传感器数量：128个超声传感器
samples = 1024   # 采样点数：每个传感器采集1024个时间点的数据

print(f"   角度数量: {angles} (不同的超声发射角度)")
print(f"   传感器数量: {elements} (超声探头上的传感器)")
print(f"   采样点数: {samples} (每个传感器的时间采样)")

# 创建模拟的超声数据 (实部)
print("\n🔬 创建模拟超声数据...")
idata = np.random.randn(angles, elements, samples) * 0.1
print(f"   idata形状: {idata.shape}")
print(f"   数据类型: {idata.dtype}")
print(f"   数据范围: [{idata.min():.3f}, {idata.max():.3f}]")

# ===== 第二部分：理解单角度 vs 多角度 =====
print("\n🎯 第二部分：单角度 vs 多角度的区别")

# 单角度重建：只用第1个角度 (索引为0)
print("   单角度重建：只使用1个发射角度")
single_angle_data = idata[0, :, :]  # 形状变成 (elements, samples)
print(f"   单角度数据形状: {single_angle_data.shape}")

# 多角度重建：使用所有75个角度
print("   多角度重建：使用全部75个发射角度")
multi_angle_data = np.mean(idata, axis=0)  # 沿着角度维度求平均
print(f"   多角度数据形状: {multi_angle_data.shape}")

# ===== 第三部分：数据转换成图像 =====
print("\n🖼️  第三部分：将超声数据转换成图像")

# 为了显示，我们截取一部分数据作为图像
img_size = 64
single_image = single_angle_data[:img_size, :img_size]
multi_image = multi_angle_data[:img_size, :img_size]

print(f"   单角度图像形状: {single_image.shape}")
print(f"   多角度图像形状: {multi_image.shape}")

# 计算图像质量差异
quality_diff = np.std(multi_image) - np.std(single_image)
print(f"   质量差异 (标准差): {quality_diff:.3f}")
print("   多角度图像通常比单角度图像有更好的对比度和清晰度")

# ===== 第四部分：可视化对比 =====
print("\n📈 第四部分：可视化单角度 vs 多角度")

plt.figure(figsize=(12, 5))

# 子图1：单角度图像
plt.subplot(1, 3, 1)
plt.imshow(single_image, cmap='gray', aspect='auto')
plt.title('单角度图像\n(质量较低，噪声多)')
plt.xlabel('采样点')
plt.ylabel('传感器')

# 子图2：多角度图像  
plt.subplot(1, 3, 2)
plt.imshow(multi_image, cmap='gray', aspect='auto')
plt.title('多角度图像\n(质量较高，噪声少)')
plt.xlabel('采样点')
plt.ylabel('传感器')

# 子图3：差异图
plt.subplot(1, 3, 3)
diff_image = multi_image - single_image
plt.imshow(diff_image, cmap='RdBu', aspect='auto')
plt.title('质量差异\n(蓝色=改善区域)')
plt.xlabel('采样点')
plt.ylabel('传感器')
plt.colorbar()

plt.tight_layout()
plt.savefig('/home/liujia/dev/AUGAN_725/Cluade/practice/step1_结果.png')
print("   图像已保存到: step1_结果.png")
plt.show()

# ===== 总结 =====
print("\n" + "="*50)
print("🎓 第一步学习总结:")
print("   1. 超声数据有3个维度：角度×传感器×采样点")
print("   2. 单角度图像：快速但质量较低")
print("   3. 多角度图像：慢速但质量较高") 
print("   4. AUGAN的目标：单角度 → 多角度的智能转换")
print("\n✅ 第一步完成！接下来运行 step2_图像重建练习.py")
print("="*50)