#!/usr/bin/env python3
"""
AUGAN学习练习 - 第二步：理解DAS图像重建算法
目标：理解如何从原始超声数据重建出图像，模拟真实的DAS算法
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

print("🎯 AUGAN学习练习 - 第二步：DAS图像重建")
print("="*50)

# ===== 第一部分：创建更真实的超声数据 =====
print("📡 第一部分：创建模拟的平面波超声数据")

# 超声成像参数 (接近真实PICMUS数据)
n_angles = 75        # 75个发射角度 (-37° 到 +37°)
n_elements = 128     # 128个接收传感器
n_samples = 2048     # 2048个时间采样点

# 创建角度数组 (弧度)
angles = np.linspace(-37*np.pi/180, 37*np.pi/180, n_angles)
print(f"   发射角度范围: {angles[0]*180/np.pi:.1f}° 到 {angles[-1]*180/np.pi:.1f}°")

# 模拟超声数据 (IQ数据: I是实部，Q是虚部)
print("   创建模拟IQ数据...")
idata = np.random.randn(n_angles, n_elements, n_samples) * 0.5
qdata = np.random.randn(n_angles, n_elements, n_samples) * 0.5

# 添加一些模拟的目标信号 (在特定位置)
for angle_idx in range(n_angles):
    # 在中心位置添加强反射信号
    center_element = n_elements // 2
    center_sample = n_samples // 2
    
    # 模拟点目标
    idata[angle_idx, center_element-5:center_element+5, center_sample-10:center_sample+10] += 2.0
    qdata[angle_idx, center_element-5:center_element+5, center_sample-10:center_sample+10] += 1.5

print(f"   IQ数据形状: I={idata.shape}, Q={qdata.shape}")

# ===== 第二部分：简化的DAS重建算法 =====
print("\n🔬 第二部分：模拟DAS (Delay-and-Sum) 重建算法")

def simple_das_reconstruction(idata, qdata, angle_indices, output_size=(256, 256)):
    """
    简化的DAS重建算法
    
    参数:
        idata, qdata: IQ数据
        angle_indices: 要使用的角度索引列表
        output_size: 输出图像尺寸
    
    返回:
        重建的图像
    """
    print(f"   使用角度数量: {len(angle_indices)}")
    
    # 选择指定角度的数据
    selected_i = idata[angle_indices, :, :]  # (选中角度, 传感器, 采样点)
    selected_q = qdata[angle_indices, :, :]
    
    # 计算复数信号幅度
    amplitude = np.sqrt(selected_i**2 + selected_q**2)
    
    # 简化的空间映射 (实际DAS算法要复杂得多)
    # 这里我们对传感器和采样点维度进行重采样
    h, w = output_size
    
    # 沿角度维度求和 (这是DAS的核心思想)
    summed_amplitude = np.sum(amplitude, axis=0)  # (传感器, 采样点)
    
    # 重采样到目标尺寸
    from scipy import ndimage
    reconstructed = ndimage.zoom(summed_amplitude, (h/n_elements, w/n_samples), mode='nearest')
    
    return reconstructed

# ===== 第三部分：单角度 vs 多角度重建对比 =====
print("\n🖼️  第三部分：对比不同角度数量的重建质量")

# 1. 单角度重建 (只用第38个角度，即中心角度0°)
center_angle = [n_angles // 2]
single_angle_img = simple_das_reconstruction(idata, qdata, center_angle)
print(f"   单角度重建完成，图像尺寸: {single_angle_img.shape}")

# 2. 少角度重建 (用5个角度)
few_angles = list(range(35, 40))  # 中心附近5个角度
few_angle_img = simple_das_reconstruction(idata, qdata, few_angles)
print(f"   少角度重建完成，使用角度: {len(few_angles)}个")

# 3. 多角度重建 (用所有75个角度)
all_angles = list(range(n_angles))
multi_angle_img = simple_das_reconstruction(idata, qdata, all_angles)
print(f"   多角度重建完成，使用角度: {len(all_angles)}个")

# ===== 第四部分：图像质量分析 =====
print("\n📊 第四部分：分析不同重建方法的图像质量")

def analyze_image_quality(img, name):
    """分析图像质量指标"""
    mean_val = np.mean(img)
    std_val = np.std(img)
    snr = mean_val / std_val if std_val > 0 else 0
    dynamic_range = np.max(img) - np.min(img)
    
    print(f"   {name}:")
    print(f"     平均值: {mean_val:.3f}")
    print(f"     标准差: {std_val:.3f}")
    print(f"     信噪比: {snr:.3f}")
    print(f"     动态范围: {dynamic_range:.3f}")
    
    return {'mean': mean_val, 'std': std_val, 'snr': snr, 'range': dynamic_range}

single_stats = analyze_image_quality(single_angle_img, "单角度重建")
few_stats = analyze_image_quality(few_angle_img, "少角度重建")
multi_stats = analyze_image_quality(multi_angle_img, "多角度重建")

# ===== 第五部分：可视化对比 =====
print("\n📈 第五部分：可视化重建结果对比")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第一行：重建图像
axes[0, 0].imshow(single_angle_img, cmap='gray', aspect='auto')
axes[0, 0].set_title(f'单角度重建\n(1个角度, SNR={single_stats["snr"]:.2f})')
axes[0, 0].set_xlabel('水平位置')
axes[0, 0].set_ylabel('深度')

axes[0, 1].imshow(few_angle_img, cmap='gray', aspect='auto')
axes[0, 1].set_title(f'少角度重建\n(5个角度, SNR={few_stats["snr"]:.2f})')
axes[0, 1].set_xlabel('水平位置')
axes[0, 1].set_ylabel('深度')

axes[0, 2].imshow(multi_angle_img, cmap='gray', aspect='auto')
axes[0, 2].set_title(f'多角度重建\n(75个角度, SNR={multi_stats["snr"]:.2f})')
axes[0, 2].set_xlabel('水平位置')
axes[0, 2].set_ylabel('深度')

# 第二行：质量对比分析
# 中心横截面对比
center_row = single_angle_img.shape[0] // 2
axes[1, 0].plot(single_angle_img[center_row, :], 'r-', label='单角度', linewidth=2)
axes[1, 0].plot(few_angle_img[center_row, :], 'g-', label='少角度', linewidth=2)
axes[1, 0].plot(multi_angle_img[center_row, :], 'b-', label='多角度', linewidth=2)
axes[1, 0].set_title('中心线横截面对比')
axes[1, 0].set_xlabel('水平位置')
axes[1, 0].set_ylabel('强度')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 质量指标柱状图
metrics = ['SNR', '动态范围', '标准差']
single_vals = [single_stats['snr'], single_stats['range']/10, single_stats['std']]
few_vals = [few_stats['snr'], few_stats['range']/10, few_stats['std']]
multi_vals = [multi_stats['snr'], multi_stats['range']/10, multi_stats['std']]

x = np.arange(len(metrics))
width = 0.25

axes[1, 1].bar(x - width, single_vals, width, label='单角度', color='red', alpha=0.7)
axes[1, 1].bar(x, few_vals, width, label='少角度', color='green', alpha=0.7)
axes[1, 1].bar(x + width, multi_vals, width, label='多角度', color='blue', alpha=0.7)
axes[1, 1].set_title('图像质量指标对比')
axes[1, 1].set_xlabel('指标类型')
axes[1, 1].set_ylabel('数值')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 角度使用示意图
angle_usage = np.zeros(n_angles)
angle_usage[center_angle] = 1  # 单角度
few_angle_usage = np.zeros(n_angles)
few_angle_usage[few_angles] = 1  # 少角度
all_angle_usage = np.ones(n_angles)  # 多角度

axes[1, 2].plot(angles*180/np.pi, angle_usage, 'r-', linewidth=3, label='单角度')
axes[1, 2].plot(angles*180/np.pi, few_angle_usage + 0.1, 'g-', linewidth=3, label='少角度')  
axes[1, 2].plot(angles*180/np.pi, all_angle_usage + 0.2, 'b-', linewidth=3, label='多角度')
axes[1, 2].set_title('使用的发射角度')
axes[1, 2].set_xlabel('角度 (度)')
axes[1, 2].set_ylabel('是否使用')
axes[1, 2].set_ylim(-0.1, 1.5)
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/liujia/dev/AUGAN_725/Cluade/practice/step2_DAS重建对比.png', dpi=150, bbox_inches='tight')
print("   对比图已保存到: step2_DAS重建对比.png")
plt.show()

# ===== 第六部分：AUGAN的作用理解 =====
print("\n" + "="*50)
print("🎓 第二步学习总结:")
print("   1. DAS算法：通过多角度数据相加提高图像质量")
print("   2. 角度越多 → 图像质量越好，但采集时间越长")
print("   3. 单角度快速但质量差，多角度慢速但质量好")
print("   4. AUGAN的创新：用AI学习单角度→多角度的映射关系")
print("   5. 这样既保持了单角度的速度，又获得了多角度的质量！")
print("\n💡 现在你理解了AUGAN要解决的核心问题:")
print("   输入：快速单角度重建图像 (质量较差)")
print("   输出：高质量多角度重建图像 (质量很好)")
print("   方法：深度学习的图像到图像翻译")
print("\n✅ 第二步完成！接下来运行 step3_简单GAN练习.py")
print("="*50)