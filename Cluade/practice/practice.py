import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # 使用 'Agg' 后端
import torch
import torch.nn as nn
angles = 75
elements = 128
samples = 1024

idata = np.random.randn(angles, elements, samples)
print(f"模拟超声数据的形状：{idata.shape}")
print(f"角度数：{angles}, 传感器数量：{elements}, 采样点数：{samples}")


print("===第二步：从数据生成图像")

single_angle_data = idata[0, :, :]
print(f"单角度数据形状：{single_angle_data.shape}")

multi_angle_data = np.mean(idata, axis=0)
print(f"多角度数据的形状：{multi_angle_data.shape}")

single_image = single_angle_data[:64, :64]
multi_image = multi_angle_data[:64, :64]

print(f"单角度图像的形状：{single_image.shape}")
print(f"多角度图像的形状：{multi_image.shape}")

# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(single_image, cmap='gray')
# plt.title('单角度图像(低质量)')

# plt.subplot(1, 2, 2)
# plt.imshow(multi_image, cmap='gray')
# plt.title('多角度图像(高质量)')
# # plt.show()
# plt.savefig('comparison_image.png')

class SimpleConverter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        
    def forward(self, x):
        return self.conv(x)


model = SimpleConverter()
fake_input = torch.randn(1, 1, 64, 64)
fake_output = model(fake_input)

print(f"输入形状：{fake_input.shape}")
print(f"输出形状：{fake_output.shape}")
