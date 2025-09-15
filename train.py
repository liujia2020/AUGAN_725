import argparse
import logging
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cubdl.example_picmus_torch import load_datasets,create_network,mk_img,dispaly_img
from options.train_options import TrainOptions
from models import create_model
from data_process import load_dataset, test_image, load_dataset_multi  # 使用原始版本避免CUDA问题
from utils.util import diagnose_network
from models.network import UnetGenerator
import math
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset,TensorDataset
from utils.metrics import image_evaluation

def makedir(opt):
    train_base = './images/' + opt.name + '/train'
    test_base = './images/' + opt.name + '/test'
    if not os.path.exists(train_base):
        os.makedirs(train_base)
    if not os.path.exists(test_base):
        os.makedirs(test_base)
    loss_path = './images/' + opt.name + '/train/loss.png'
    return loss_path

if __name__ == '__main__':
    print("🚀 启动训练脚本...")
    
    # Initial setting and corresponding parameters
    plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")
    das, iqdata, xlims, zlims = create_network(plane_wave_data, [1])

    opt = TrainOptions().parse()
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load the model
    model = create_model(opt)
    model.setup(opt)
    print(f"✅ 模型已加载到设备: {model.device}")
    
    total_iters = 0
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    loss_path = makedir(opt)

    print("📂 开始加载数据集...")
    start_load_time = time.time()
    
    # 使用原始数据加载（更稳定）
    # img_dataset = load_dataset(opt, opt.phase, 0)
    
    img_dataset = load_dataset_multi(opt, opt.phase, 0)  # 🎯 使用多数据集版本
    print("数据集样本总数：", len(img_dataset))
    dataset_len = img_dataset.len
    
    load_time = time.time() - start_load_time
    print(f"✅ 数据集加载完成，用时: {load_time:.2f}秒")
    print(f"📊 数据集大小: {dataset_len}")
    
    # 🔥 关键优化：DataLoader配置
    available_workers = os.cpu_count() if os.cpu_count() else 4
    num_workers = min(8, available_workers)  # 增加到8个worker
    
    print(f"⚙️ 系统有 {available_workers} 个CPU核心，使用 {num_workers} 个worker")
    
    train_loader = DataLoader(
        dataset=img_dataset, 
        num_workers=num_workers,               # 🔥 关键：8个并行worker
        batch_size=1,                          # 保持batch_size=1避免内存问题
        shuffle=True,
        pin_memory=torch.cuda.is_available(),  # 🔥 关键：内存固定
        prefetch_factor=8,                     # 🔥 关键：大幅增加预取
        persistent_workers=True,               # 🔥 关键：保持worker进程
        drop_last=False
    )
    
    print(f"🔥 高性能DataLoader配置:")
    print(f"   - Workers: {num_workers}")
    print(f"   - Pin Memory: {torch.cuda.is_available()}")
    print(f"   - Prefetch Factor: 8")
    print(f"   - Persistent Workers: True")
    
    # 初始化损失数组
    lossG = np.zeros(opt.n_epochs + opt.niter_decay)
    lossD = np.zeros(opt.n_epochs + opt.niter_decay)

    # Image evaluation
    img_eva = image_evaluation()

    print("🎓 开始训练...")
    
    # 性能统计
    batch_times = []
    gpu_utilizations = []
    
    # Training process
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        
        # 详细的进度条
        train_bar = tqdm(
            train_loader, 
            desc=f'Epoch {epoch}/{opt.n_epochs + opt.niter_decay}',
            ncols=140,
            unit='batch',
            smoothing=0.1
        )
        
        epoch_batch_times = []
        
        for batch_idx, batch_data in enumerate(train_bar):
            batch_start = time.time()
            
            # 从字典中获取数据
            data = batch_data['A']
            target = batch_data['B']
            
            # 训练
            model.set_input(data, target)
            model.optimize_parameters()
            
            epoch_iter += opt.batch_size
            batch_time = time.time() - batch_start
            epoch_batch_times.append(batch_time)
            
            # 实时性能监控
            if batch_idx % 20 == 0:
                try:
                    performance_info = {
                        'Batch_ms': f"{batch_time*1000:.0f}ms"
                    }
                    
                    # GPU信息
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1e9
                        gpu_cached = torch.cuda.memory_reserved() / 1e9
                        performance_info.update({
                            'GPU_mem': f"{gpu_memory:.1f}GB",
                            'GPU_cache': f"{gpu_cached:.1f}GB"
                        })
                    
                    # 损失信息
                    try:
                        if hasattr(model, 'loss_G_GAN') and hasattr(model, 'loss_D_real'):
                            performance_info.update({
                                'G_loss': f"{float(model.loss_G_GAN):.3f}",
                                'D_loss': f"{float(model.loss_D_real):.3f}"
                            })
                    except:
                        pass
                    
                    train_bar.set_postfix(performance_info)
                    
                except ImportError:
                    # 如果模块不可用，显示基本信息
                    train_bar.set_postfix({
                        'Time': f"{batch_time*1000:.0f}ms",
                        'Status': 'Training...'
                    })
                except Exception as e:
                    # 任何其他错误，保持简单显示
                    train_bar.set_postfix({'Status': 'Running...'})
        
        # Epoch统计
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(epoch_batch_times) if epoch_batch_times else 0
        batches_per_sec = len(train_loader) / epoch_time if epoch_time > 0 else 0
        
        print(f"\n📈 Epoch {epoch} 完成:")
        print(f"   ⏱️  总时间: {epoch_time:.1f}秒")
        print(f"   📊 平均批次时间: {avg_batch_time*1000:.1f}ms")
        print(f"   🚀 处理速度: {batches_per_sec:.1f} batches/秒")
        
        # 保存性能数据
        batch_times.extend(epoch_batch_times)
        
        # 损失记录
        try:
            if hasattr(model, 'loss_G_GAN'):
                lossG[epoch-1] = float(model.loss_G_GAN)
            if hasattr(model, 'loss_D_real'):
                lossD[epoch-1] = float(model.loss_D_real)
        except:
            pass
        
        # 定期保存和清理
        if epoch % 10 == 0:
            model.save_networks(epoch)
            print(f"💾 模型已保存 - Epoch {epoch}")
            
            # GPU内存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                current_memory = torch.cuda.memory_allocated() / 1e9
                print(f"🧹 GPU缓存已清理，当前使用: {current_memory:.1f}GB")
        
        # 更新学习率
        model.update_learning_rate()
        
        print("-" * 60)

    print("\n🎉 训练完成！")
    
    # 保存最终模型
    model.save_networks('latest')
    print("💾 最终模型已保存")
    
    # 性能报告
    if batch_times:
        avg_batch_time = np.mean(batch_times) * 1000
        min_batch_time = np.min(batch_times) * 1000
        max_batch_time = np.max(batch_times) * 1000
        
        print(f"\n📊 性能报告:")
        print(f"   平均批次时间: {avg_batch_time:.1f}ms")
        print(f"   最快批次时间: {min_batch_time:.1f}ms")
        print(f"   最慢批次时间: {max_batch_time:.1f}ms")
        print(f"   性能优化效果: 多进程数据加载 + 内存固定 + 预取缓冲")
        
        if avg_batch_time < 200:
            print("✅ 性能优秀！批次处理时间 < 200ms")
        elif avg_batch_time < 500:
            print("⚠️  性能良好，但仍有优化空间")
        else:
            print("❌ 性能需要进一步优化")