# Semantic Segmentation with PSPNet

> Georgia Tech CS-6476 Computer Vision

## Overview

This project implements semantic segmentation using the Pyramid Scene Parsing Network (PSPNet). The model performs pixel-wise classification on driving scene images, labeling each pixel as road, car, pedestrian, building, sky, etc. Includes transfer learning from CamVid to KITTI dataset.

## Key Implementations

| File | Description |
|------|-------------|
| `part1_ppm.py` | Pyramid Pooling Module (PPM) — multi-scale context aggregation |
| `part2_dataset.py` | Custom dataset loader for semantic segmentation |
| `part3_training_utils.py` | Training loop, loss functions, learning rate scheduling |
| `part4_segmentation_net.py` | Base segmentation network architecture |
| `part5_pspnet.py` | Full PSPNet implementation with ResNet backbone |
| `part6_transfer_learning.py` | Domain adaptation from CamVid to KITTI |
| `iou.py` | Intersection over Union (IoU/mIoU) metrics |

## Architecture

**PSPNet** combines:
1. **ResNet-50 backbone** — Feature extraction (pretrained on ImageNet)
2. **Pyramid Pooling Module** — Multi-scale context via pooling at 1×1, 2×2, 3×3, 6×6
3. **Decoder** — Upsampling to original resolution with skip connections

The PPM captures both local details and global context, crucial for understanding scene layout.

## Trained Models

Trained weights are in `exp/`:
- `exp/camvid/PSPNet/model/` — Trained on CamVid (11 classes)
- `exp/kitti/PSPNet/model/` — Transfer learned on KITTI

## Datasets

- **CamVid**: 701 driving scene images, 11 semantic classes
- **KITTI**: Autonomous driving benchmark, adapted via transfer learning

Dataset lists are in `src/dataset_lists/`. Full images should be downloaded separately.

## Project Structure

```
├── src/vision/          # PSPNet implementation
├── src/dataset_lists/   # Train/val/test splits
├── exp/                 # Trained model weights
├── doc/                 # Project report
├── tests/               # Unit tests
├── proj4_local.ipynb    # Local training notebook
└── proj4_colab.ipynb    # Google Colab notebook (GPU)
```

## Setup

```bash
conda env create -f conda/environment.yml
conda activate cv_proj4
pip install -e .
```

## Usage

```python
from vision.part5_pspnet import PSPNet

# Initialize model
model = PSPNet(num_classes=11, pretrained=True)

# Load trained weights
model.load_state_dict(torch.load('exp/camvid/PSPNet/model/train_epoch_50.pth'))

# Inference
segmentation_mask = model(image_tensor)
```

## Metrics

Model performance is evaluated using mean Intersection over Union (mIoU) across all semantic classes.
