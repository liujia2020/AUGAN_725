# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AUGAN (Attention U-Net based Generative Adversarial Network) is a deep learning project for ultrasound image enhancement using generative adversarial networks. The project builds upon the CUBDL (Challenge on Ultrasound Beamforming with Deep Learning) framework to improve ultrasound image quality through learned image-to-image translation.

### Core Purpose
- Transform single-angle ultrasound reconstructions (low quality) into multi-angle composite images (high quality)
- Uses Pix2Pix GAN architecture with U-Net generator and PatchGAN discriminator
- Incorporates VGG-based perceptual loss for enhanced visual quality

## Development Commands

### Training
```bash
# Basic training command
python train.py --name experiment_name --model pix2pix --netG unet_256 --netD basic

# Training with specific parameters
python train.py --name unet_b002 --model pix2pix --netG unet_256 --netD basic --n_epochs 100 --lr 0.0002 --batch_size 1

# Continue training from checkpoint
python train.py --continue_train --epoch_count 50 --name experiment_name
```

### Testing/Inference
```bash
# Basic testing command
python test.py --name experiment_name --model test --netG unet_256

# Test with specific number of samples
python test.py --name experiment_name --model test --netG unet_256 --num_test 100

# Test specific epoch
python test.py --name experiment_name --epoch 100 --model test --netG unet_256
```

### Data Processing
The project uses PICMUS ultrasound datasets. Data loading is handled automatically through:
- `data_process.py`: Main data loading and preprocessing
- `cubdl/`: Contains PICMUS dataset utilities and DAS reconstruction algorithms

## Architecture Overview

### Model Architecture
- **Base Framework**: Pix2Pix conditional GAN
- **Generator**: U-Net with 256x256 resolution (8 layers)
- **Discriminator**: PatchGAN (70x70 receptive field)
- **Loss Functions**: 
  - GAN loss (adversarial training)
  - L2 loss (pixel-level supervision) 
  - VGG perceptual loss (high-level feature matching)

### Key Components

#### Core Training Files
- `train.py`: Main training script with performance optimizations
- `test.py`: Inference and evaluation script  
- `models/pix2pix_model.py`: Core GAN model implementation
- `models/base_model.py`: Abstract base class for all models
- `models/network.py`: Network architectures and utilities

#### Configuration System
- `options/base_options.py`: Shared configuration parameters
- `options/train_options.py`: Training-specific options
- `options/test_options.py`: Testing-specific options

#### Data Processing
- `data_process.py`: Dataset loading and preprocessing
- `cubdl/`: PICMUS dataset utilities and beamforming algorithms
  - `PlaneWaveData.py`: Plane wave acquisition data structure
  - `PixelGrid.py`: Reconstruction grid definition
  - `das_torch.py`: Delay-and-sum beamforming implementation
  - `metrics.py`: Evaluation metrics (CNR, gCNR, SNR, PSNR, etc.)

### File Structure Logic

```
AUGAN_725/
├── models/           # Neural network architectures
├── options/          # Configuration management
├── utils/            # Utilities (image pool, SSIM, etc.)
├── cubdl/           # PICMUS dataset and beamforming
├── data/            # Raw data storage
├── img_data1/       # Processed PICMUS datasets (.mat files)
├── checkpoints/     # Saved model weights
├── results/         # Test outputs and evaluation
├── train.py         # Training script
└── test.py          # Testing script
```

## Development Workflow

### Setting Up Training
1. Ensure PICMUS datasets are in `img_data1/` directory
2. Configure training options in `options/train_options.py` or via command line
3. Run training: `python train.py --name your_experiment`
4. Monitor training through console output and saved loss plots

### Model Evaluation  
1. Ensure trained model exists in `checkpoints/experiment_name/`
2. Run inference: `python test.py --name experiment_name`
3. Results saved to `images/experiment_name/test/` and `results/experiment_name/`

### Adding New Features
- Extend `BaseModel` for new model architectures
- Modify `network.py` for new network components
- Update options classes for new configuration parameters
- Use the existing data loading pipeline in `data_process.py`

## Important Implementation Details

### Memory Management
- Training uses DataLoader with 8 workers and memory pinning for GPU efficiency
- Gradient computation controlled via `set_requires_grad()` to optimize memory usage
- Regular GPU cache clearing during training

### Model Saving/Loading
- Models automatically saved every 10 epochs to `checkpoints/`
- Use `--continue_train` flag to resume from checkpoints
- Latest model saved as 'latest_net_G.pth' and 'latest_net_D.pth'

### Performance Optimizations
- Multi-worker data loading with persistent workers
- Memory pinning for faster GPU transfers  
- Prefetch factor of 8 for reduced data loading bottlenecks
- Mixed precision training support (if configured)

### Loss Function Design
The model uses a composite loss:
```
Total Loss = λ_GAN * L_GAN + λ_L1 * L_L2 + λ_content * L_content
```
- `L_GAN`: Adversarial loss for realistic image generation
- `L_L2`: Pixel-wise L2 loss (not L1 despite parameter name)
- `L_content`: VGG feature-based perceptual loss

### Dataset Requirements
- PICMUS format ultrasound data (.mat files)
- Plane wave acquisitions with multiple angles
- IQ (In-phase/Quadrature) demodulated data
- Requires specific file naming convention in `img_data1/`

### Hardware Requirements
- CUDA-compatible GPU recommended for training
- Minimum 8GB GPU memory for batch_size=1 with 256x256 images
- CPU fallback available but significantly slower

This codebase implements a production-ready ultrasound image enhancement system with comprehensive training, testing, and evaluation capabilities.