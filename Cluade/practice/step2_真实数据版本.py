#!/usr/bin/env python3
"""
AUGAN学习练习 - 第二步：使用真实PICMUS数据的DAS重建
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 导入AUGAN项目的模块
sys.path.append('/home/liujia/dev/AUGAN_725')
from cubdl.example_picmus_torch import load_datasets, create_network, mk_img

print("🎯 使用真实PICMUS数据进行DAS重建练习")
print("="*50)

# ===== 第一步：加载真实PICMUS数据 =====
print("📡 第一步：加载真实PICMUS数据")

# 加载数据（和训练脚本使用的完全一样）
plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")

print(f"   数据类型: {type(plane_wave_data)}")
print(f"   发射角度数: {len(plane_wave_data.angles)}")
print(f"   IQ数据形状: {plane_wave_data.idata.shape}")
print(f"   传感器位置: {plane_wave_data.ele_pos.shape}")
print(f"   中心频率: {plane_wave_data.fc/1e6:.1f} MHz")
print(f"   采样频率: {plane_wave_data.fs/1e6:.1f} MHz")

# ===== 第二步：创建不同角度的DAS网络 =====
print("\n🔬 第二步：创建DAS重建网络")

# 1. 单角度网络（第38个角度，约0度）
single_angle = [37]  # 中心角度
das_single, iqdata, xlims, zlims = create_network(plane_wave_data, single_angle)
print(f"   单角度DAS网络创建完成，使用角度: {single_angle}")

# 2. 少角度网络（5个角度）
few_angles = [35, 36, 37, 38, 39]  # 中心附近5个角度
das_few, _, _, _ = create_network(plane_wave_data, few_angles)
print(f"   少角度DAS网络创建完成，使用角度: {len(few_angles)}个")

# 3. 多角度网络（所有75个角度）
all_angles = list(range(len(plane_wave_data.angles)))
das_multi, _, _, _ = create_network(plane_wave_data, all_angles)
print(f"   多角度DAS网络创建完成，使用角度: {len(all_angles)}个")

print(f"   成像区域: X=[{xlims[0]*1000:.1f}, {xlims[1]*1000:.1f}]mm")
print(f"   成像区域: Z=[{zlims[0]*1000:.1f}, {zlims[1]*1000:.1f}]mm")

# ===== 第三步：执行DAS重建 =====
print("\n🖼️  第三步：执行真实DAS图像重建")

# 使用真实的mk_img函数重建图像
print("   重建单角度图像...")
single_img = mk_img(das_single, iqdata)

print("   重建少角度图像...")
few_img = mk_img(das_few, iqdata)

print("   重建多角度图像...")
multi_img = mk_img(das_multi, iqdata)

print(f"   单角度图像形状: {single_img.shape}")
print(f"   少角度图像形状: {few_img.shape}")
print(f"   多角度图像形状: {multi_img.shape}")

# ===== 第四步：图像质量分析 =====
print("\n📊 第四步：分析真实数据的图像质量")

def analyze_ultrasound_quality(img, name):
    """分析超声图像质量"""
    # 转换为dB显示
    img_db = 20 * np.log10(np.abs(img) + 1e-10)
    img_db -= np.max(img_db)  # 归一化到0dB
    
    mean_val = np.mean(img_db)
    std_val = np.std(img_db)
    dynamic_range = np.max(img_db) - np.min(img_db)
    
    print(f"   {name}:")
    print(f"     平均强度: {mean_val:.1f} dB")
    print(f"     标准差: {std_val:.1f} dB")
    print(f"     动态范围: {dynamic_range:.1f} dB")
    
    return img_db, {'mean': mean_val, 'std': std_val, 'range': dynamic_range}

single_db, single_stats = analyze_ultrasound_quality(single_img, "单角度重建")
few_db, few_stats = analyze_ultrasound_quality(few_img, "少角度重建")
multi_db, multi_stats = analyze_ultrasound_quality(multi_img, "多角度重建（目标质量）")

# ===== 第五步：可视化真实超声图像 =====
print("\n📈 第五步：可视化真实超声重建结果")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 设置显示参数
extent = [xlims[0]*1000, xlims[1]*1000, zlims[1]*1000, zlims[0]*1000]
vmin, vmax = -60, 0  # dB范围

# 第一行：超声图像显示
axes[0, 0].imshow(single_db, vmin=vmin, vmax=vmax, cmap='gray', extent=extent, aspect='auto')
axes[0, 0].set_title(f'单角度重建\n(输入图像类型)')
axes[0, 0].set_xlabel('横向位置 (mm)')
axes[0, 0].set_ylabel('深度 (mm)')

axes[0, 1].imshow(few_db, vmin=vmin, vmax=vmax, cmap='gray', extent=extent, aspect='auto')
axes[0, 1].set_title(f'少角度重建\n({len(few_angles)}个角度)')
axes[0, 1].set_xlabel('横向位置 (mm)')
axes[0, 1].set_ylabel('深度 (mm)')

im = axes[0, 2].imshow(multi_db, vmin=vmin, vmax=vmax, cmap='gray', extent=extent, aspect='auto')
axes[0, 2].set_title(f'多角度重建\n(目标图像类型，{len(all_angles)}个角度)')
axes[0, 2].set_xlabel('横向位置 (mm)')
axes[0, 2].set_ylabel('深度 (mm)')

# 添加颜色条
plt.colorbar(im, ax=axes[0, 2], label='强度 (dB)')

# 第二行：质量分析
# 横向轮廓对比
center_depth = single_db.shape[0] // 2
x_axis = np.linspace(xlims[0]*1000, xlims[1]*1000, single_db.shape[1])

axes[1, 0].plot(x_axis, single_db[center_depth, :], 'r-', linewidth=2, label='单角度')
axes[1, 0].plot(x_axis, few_db[center_depth, :], 'g-', linewidth=2, label='少角度')
axes[1, 0].plot(x_axis, multi_db[center_depth, :], 'b-', linewidth=2, label='多角度')
axes[1, 0].set_title('中心深度横向轮廓对比')
axes[1, 0].set_xlabel('横向位置 (mm)')
axes[1, 0].set_ylabel('强度 (dB)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(-60, 0)

# 纵向轮廓对比
center_lateral = single_db.shape[1] // 2
z_axis = np.linspace(zlims[0]*1000, zlims[1]*1000, single_db.shape[0])

axes[1, 1].plot(single_db[:, center_lateral], z_axis, 'r-', linewidth=2, label='单角度')
axes[1, 1].plot(few_db[:, center_lateral], z_axis, 'g-', linewidth=2, label='少角度')
axes[1, 1].plot(multi_db[:, center_lateral], z_axis, 'b-', linewidth=2, label='多角度')
axes[1, 1].set_title('中心横向纵向轮廓对比')
axes[1, 1].set_xlabel('强度 (dB)')
axes[1, 1].set_ylabel('深度 (mm)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim(-60, 0)
axes[1, 1].invert_yaxis()

# 质量指标对比
metrics = ['平均强度', '标准差', '动态范围']
single_vals = [single_stats['mean'], single_stats['std'], single_stats['range']/10]
few_vals = [few_stats['mean'], few_stats['std'], few_stats['range']/10]
multi_vals = [multi_stats['mean'], multi_stats['std'], multi_stats['range']/10]

x = np.arange(len(metrics))
width = 0.25

axes[1, 2].bar(x - width, np.abs(single_vals), width, label='单角度', color='red', alpha=0.7)
axes[1, 2].bar(x, np.abs(few_vals), width, label='少角度', color='green', alpha=0.7)
axes[1, 2].bar(x + width, np.abs(multi_vals), width, label='多角度', color='blue', alpha=0.7)
axes[1, 2].set_title('图像质量指标对比')
axes[1, 2].set_xlabel('指标类型')
axes[1, 2].set_ylabel('数值 (绝对值)')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(metrics)
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/liujia/dev/AUGAN_725/Cluade/practice/step2_真实PICMUS数据重建.png', dpi=150, bbox_inches='tight')
print("   真实数据重建图已保存到: step2_真实PICMUS数据重建.png")
plt.show()

# ===== 总结 =====
print("\n" + "="*50)
print("🎓 真实数据版本总结:")
print("   1. 使用了项目中的真实PICMUS超声数据")
print("   2. 看到了真实的超声图像质量差异")
print("   3. 单角度图像确实比多角度图像质量差")
print("   4. 这就是AUGAN要学习的输入→输出映射！")
print(f"\n📏 真实数据规模:")
print(f"   - 图像尺寸: {single_img.shape}")
print(f"   - 成像范围: {xlims[1]-xlims[0]:.3f}m × {zlims[1]-zlims[0]:.3f}m")
print(f"   - 发射角度: {len(plane_wave_data.angles)}个")
print("="*50)