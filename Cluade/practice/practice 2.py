import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('/home/liujia/dev/AUGAN_725')
from cubdl.example_picmus_torch import load_datasets, create_network, mk_img

plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")

print(plane_wave_data)

print(f"数据类型：{type(plane_wave_data)}")
print(f"发射角度数：{len(plane_wave_data.angles)}")
print(f"IQ数据形状：{plane_wave_data.idata.shape}")
print(f"传感器位置：{plane_wave_data.ele_pos.shape}")
print(f"中心频率：{plane_wave_data.fc/1e6:.1f} MHz")
print(F"采样频率：{plane_wave_data.fs/1e6:.1f}")


single_angle = [37]
das_single, iqdata, xlims, zlims = create_network(plane_wave_data, single_angle)
print(f"单角度DAS网络创建完成，使用角度：{single_angle}")

few_angle = [35, 36, 37, 38, 39]
das_few, _, _, _ = create_network(plane_wave_data, few_angle)
print(f"少角度DAS网络创建完成，使用角度：{len(few_angle)}")

all_angle = list(range(len(plane_wave_data.angles)))
das_multi, _, _, _ = create_network(plane_wave_data, all_angle)
print(f"多角度DAS网络创建完成，使用角度：{len(all_angle)}")

print(f"成像区域：X=[{xlims[0]*1000:.1f}, {xlims[1]*1000:.1f}]mm")
print(f"成像区域：Z=[{xlims[0]*1000:.1f}, {xlims[1]*1000:.1f}]mm")