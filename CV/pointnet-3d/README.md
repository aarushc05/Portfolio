# 3D Point Cloud Classification with PointNet

> Georgia Tech CS-6476 Computer Vision

## Overview

This project implements PointNet for classifying 3D point clouds directly, without converting to voxels or images. The architecture learns features from raw (x, y, z) coordinates and achieves permutation invariance through symmetric functions. Includes the T-Net spatial transformer for improved robustness.

## Key Implementations

| File | Description |
|------|-------------|
| `part1_dataloader.py` | Point cloud data loading and preprocessing |
| `part2_baseline.py` | Simple baseline using global max pooling |
| `part3_pointnet.py` | Core PointNet architecture |
| `part4_analysis.py` | Visualization and analysis of learned features |
| `part5_tnet.py` | T-Net spatial transformer network |
| `training.py` | Training loop and utilities |

## Architecture

**PointNet** key innovations:
1. **Point-wise MLPs** — Shared weights process each point independently
2. **Symmetric Function** — Max pooling aggregates point features (permutation invariant)
3. **T-Net** — Learned 3×3 transformation for input alignment

```
Input Points (N×3) → T-Net → MLP(64→128→1024) → MaxPool → MLP(512→256→K) → Classes
```

The T-Net predicts a transformation matrix to canonicalize the input point cloud orientation, making the model robust to rigid transformations.

## Trained Models

Pre-trained weights are in `output/`:
- `Baseline.pt` — Simple max-pooling baseline
- `PointNet.pt` — Full PointNet without T-Net
- `PointNetTNet.pt` — PointNet with spatial transformer

## Dataset

The model is trained on 3D LiDAR point cloud sweeps. Each sample contains (x, y, z) coordinates for thousands of points representing a scene.

## Project Structure

```
├── src/vision/          # PointNet implementation
├── output/              # Trained model weights
├── docs/                # Project report
├── tests/               # Unit tests
└── proj5.ipynb          # Training and evaluation notebook
```

## Setup

```bash
conda env create -f conda/environment.yml
conda activate cv_proj5
pip install -e .
```

## Usage

```python
from vision.part3_pointnet import PointNet
from vision.part5_tnet import PointNetTNet

# Load PointNet with T-Net
model = PointNetTNet(num_classes=10)
model.load_state_dict(torch.load('output/PointNetTNet.pt'))

# Inference on point cloud (B × N × 3)
predictions = model(point_cloud_tensor)
```

## Key Insights

- **Permutation Invariance**: Max pooling ensures output is independent of point ordering
- **T-Net Regularization**: Orthogonality loss keeps transformation close to orthogonal
- **Critical Points**: Analysis reveals which points most influence the global feature

## References

- [PointNet: Deep Learning on Point Sets (Qi et al., 2017)](https://arxiv.org/abs/1612.00593)
