#!/usr/bin/env python3
"""
AUGAN 测试脚本 - 详细注释版
test_annotated.py - 使用训练好的模型对测试数据进行推理和评估

主要功能:
1. 加载训练好的AUGAN模型
2. 对测试数据进行推理生成
3. 计算各种评估指标
4. 保存测试结果和可视化图像

测试流程:
模型加载 → 数据准备 → 逐张推理 → 指标计算 → 结果保存
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入AUGAN项目模块
from options.test_options import TestOptions
from models import create_model
from data_process import load_dataset, test_image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils.metrics import image_evaluation
from cubdl.example_picmus_torch import load_datasets, create_network, mk_img, dispaly_img


def setup_test_environment():
    """
    设置测试环境
    
    返回:
        device: 计算设备
    
    功能:
        - 检测GPU/CPU可用性
        - 显示设备信息
        - 设置测试模式
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 测试设备: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💾 GPU内存: {gpu_memory:.1f} GB")
    
    return device


def load_test_model(opt):
    """
    加载训练好的模型
    
    参数:
        opt: 测试选项配置
    
    返回:
        model: 加载的模型对象
    
    功能:
        1. 创建模型架构
        2. 加载预训练权重
        3. 设置为评估模式
    """
    print("🏗️  加载训练好的模型...")
    
    # 创建模型
    model = create_model(opt)
    model.setup(opt)
    
    # 设置为评估模式（关闭dropout、batchnorm等）
    model.eval()
    
    print(f"✅ 模型加载完成: {type(model).__name__}")
    print(f"📁 模型路径: ./checkpoints/{opt.name}/")
    
    return model


def prepare_test_data(opt):
    """
    准备测试数据
    
    参数:
        opt: 测试选项配置
    
    返回:
        img_dataset: 测试数据集
        dataset_len: 数据集大小
        das, iqdata, xlims, zlims: PICMUS相关数据
    
    功能:
        1. 加载PICMUS测试数据
        2. 创建DAS网络
        3. 准备数据集迭代器
    """
    print("📡 准备测试数据...")
    
    # 加载PICMUS数据集
    plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")
    das, iqdata, xlims, zlims = create_network(plane_wave_data, [1])
    
    # 加载测试数据集
    print("📂 加载测试数据集...")
    img_dataset = load_dataset(opt, opt.phase, 0)
    dataset_len = img_dataset.len
    
    print(f"📊 测试数据集大小: {dataset_len} 个样本")
    
    return img_dataset, dataset_len, das, iqdata, xlims, zlims


def run_inference(model, img_dataset, dataset_len, opt, xlims, zlims):
    """
    执行模型推理
    
    参数:
        model: 训练好的模型
        img_dataset: 测试数据集
        dataset_len: 数据集大小
        opt: 测试选项
        xlims, zlims: 成像区域范围
    
    返回:
        results: 推理结果字典
    
    功能:
        1. 逐张图像进行推理
        2. 收集生成结果
        3. 计算评估指标
        4. 保存可视化结果
    """
    print("🔬 开始模型推理...")
    
    # 初始化结果存储
    results = {
        'input_images': [],
        'generated_images': [],
        'target_images': [],
        'metrics': {
            'ssim': [],
            'psnr': [],
            'mse': []
        }
    }
    
    # 创建结果保存目录
    results_dir = f'./results/{opt.name}'
    os.makedirs(results_dir, exist_ok=True)
    
    # 推理循环
    inference_start_time = time.time()
    
    # 限制测试数量（避免过长时间）
    test_samples = min(dataset_len, opt.num_test)
    print(f"🧪 将测试 {test_samples} 个样本")
    
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for i in tqdm(range(test_samples), desc="推理进度"):
            try:
                # 获取测试数据
                data = img_dataset.get_item(i)
                input_img = data['A']
                target_img = data['B']
                
                # 设置模型输入
                model.set_input(data)
                
                # 执行前向传播
                model.test()  # 运行模型推理
                
                # 获取生成结果
                generated_img = model.fake_B
                
                # 转换为numpy数组用于评估
                input_np = input_img.detach().cpu().numpy()
                generated_np = generated_img.detach().cpu().numpy()
                target_np = target_img.detach().cpu().numpy()
                
                # 存储结果
                results['input_images'].append(input_np)
                results['generated_images'].append(generated_np)
                results['target_images'].append(target_np)
                
                # 计算评估指标
                metrics = calculate_metrics(generated_img, target_img)
                for key, value in metrics.items():
                    results['metrics'][key].append(value)
                
                # 定期保存可视化结果
                if i % 10 == 0 or i < 20:  # 前20张和每10张保存一次
                    save_visualization(input_img, generated_img, target_img,
                                     xlims, zlims, i, opt.name, results_dir)
                
            except Exception as e:
                print(f"⚠️  样本 {i} 推理失败: {e}")
                continue
    
    inference_time = time.time() - inference_start_time
    print(f"⏱️  推理完成，用时: {inference_time:.2f}秒")
    print(f"⚡ 平均推理速度: {test_samples/inference_time:.2f} 样本/秒")
    
    return results


def calculate_metrics(generated_img, target_img):
    """
    计算图像质量评估指标
    
    参数:
        generated_img: 生成图像
        target_img: 目标图像
    
    返回:
        metrics: 评估指标字典
    
    指标说明:
        - SSIM: 结构相似性指数 [0,1]，越大越好
        - PSNR: 峰值信噪比，越大越好
        - MSE: 均方误差，越小越好
    """
    try:
        # 确保图像在CPU上进行计算
        gen_cpu = generated_img.detach().cpu()
        tar_cpu = target_img.detach().cpu()
        
        # 计算MSE
        mse = torch.mean((gen_cpu - tar_cpu) ** 2).item()
        
        # 计算PSNR
        if mse > 0:
            psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(torch.tensor(mse))).item()
        else:
            psnr = float('inf')
        
        # 计算SSIM（使用utils中的SSIM函数）
        try:
            from utils.pytorch_ssim import ssim
            ssim_value = ssim(gen_cpu, tar_cpu).item()
        except:
            ssim_value = 0.0  # 如果SSIM计算失败，设为0
        
        return {
            'ssim': ssim_value,
            'psnr': psnr,
            'mse': mse
        }
    
    except Exception as e:
        print(f"⚠️  指标计算失败: {e}")
        return {'ssim': 0.0, 'psnr': 0.0, 'mse': float('inf')}


def save_visualization(input_img, generated_img, target_img, xlims, zlims, 
                      idx, exp_name, results_dir):
    """
    保存可视化结果
    
    参数:
        input_img: 输入图像（单角度DAS）
        generated_img: 生成图像（AUGAN输出）
        target_img: 目标图像（多角度复合）
        xlims, zlims: 成像区域范围
        idx: 图像索引
        exp_name: 实验名称
        results_dir: 结果保存目录
    
    功能:
        使用PICMUS标准显示函数生成三图对比
    """
    try:
        # 调用原有的可视化函数
        test_image(input_img, generated_img, target_img, 
                  xlims, zlims, idx, 'test', exp_name)
        
        # 另外保存到results目录
        save_path = os.path.join(results_dir, f'result_{idx:03d}.png')
        if os.path.exists(f'./images/{exp_name}/test/{idx}_test.png'):
            import shutil
            shutil.copy(f'./images/{exp_name}/test/{idx}_test.png', save_path)
    
    except Exception as e:
        print(f"⚠️  可视化保存失败 (样本 {idx}): {e}")


def print_final_statistics(results):
    """
    打印最终统计结果
    
    参数:
        results: 推理结果字典
    
    功能:
        计算并显示各种评估指标的统计信息
    """
    print("\n" + "="*50)
    print("📊 测试结果统计")
    print("="*50)
    
    metrics = results['metrics']
    
    for metric_name, values in metrics.items():
        if values:  # 确保有数据
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            print(f"{metric_name.upper()}:")
            print(f"  平均值: {mean_val:.4f} ± {std_val:.4f}")
            print(f"  范围: [{min_val:.4f}, {max_val:.4f}]")
            print()
    
    print(f"✅ 总测试样本数: {len(results['generated_images'])}")
    print("="*50)


def save_results(results, opt):
    """
    保存测试结果到文件
    
    参数:
        results: 推理结果
        opt: 测试选项
    
    功能:
        将评估指标保存为文本文件，便于后续分析
    """
    results_dir = f'./results/{opt.name}'
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存评估指标
    metrics_file = os.path.join(results_dir, 'metrics.txt')
    
    with open(metrics_file, 'w') as f:
        f.write("AUGAN 测试结果\n")
        f.write("="*50 + "\n")
        f.write(f"实验名称: {opt.name}\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试样本数: {len(results['generated_images'])}\n\n")
        
        for metric_name, values in results['metrics'].items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"{metric_name.upper()}: {mean_val:.4f} ± {std_val:.4f}\n")
    
    print(f"📁 结果已保存到: {metrics_file}")


def main():
    """
    主函数 - 测试程序入口
    
    完整测试流程:
    1. 初始化测试环境
    2. 解析测试选项
    3. 加载模型和数据
    4. 执行推理和评估
    5. 保存结果和统计
    """
    print("🧪 AUGAN测试程序启动...")
    print("="*50)
    
    # 1. 设置测试环境
    device = setup_test_environment()
    
    # 2. 解析测试选项
    print("📋 解析测试配置...")
    opt = TestOptions().parse()
    
    # 3. 加载模型
    model = load_test_model(opt)
    
    # 4. 准备测试数据
    img_dataset, dataset_len, das, iqdata, xlims, zlims = prepare_test_data(opt)
    
    # 5. 执行推理
    results = run_inference(model, img_dataset, dataset_len, opt, xlims, zlims)
    
    # 6. 显示统计结果
    print_final_statistics(results)
    
    # 7. 保存结果
    save_results(results, opt)
    
    # 8. 完成总结
    print("🎉 测试完成！")
    print(f"📁 可视化结果: ./images/{opt.name}/test/")
    print(f"📊 评估结果: ./results/{opt.name}/")
    print("="*50)


if __name__ == '__main__':
    """
    程序入口点
    
    使用方法:
        python test_annotated.py --name experiment_name --model test --netG unet_128
    
    常用参数:
        --name: 实验名称（对应训练时的名称）
        --model: 模型类型 (test)
        --netG: 生成器架构（与训练时一致）
        --num_test: 测试样本数量
        --results_dir: 结果保存目录
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"❌ 测试出现错误: {e}")
        import traceback
        traceback.print_exc()