# Scene Recognition with Deep Learning

> Georgia Tech CS-6476 Computer Vision

## Overview

This project implements scene classification using convolutional neural networks, progressing from a simple custom CNN to transfer learning with ResNet18. The models classify images into 15 scene categories (bedroom, kitchen, forest, highway, etc.) and also perform multi-label attribute prediction.

## Key Implementations

| File | Description |
|------|-------------|
| `simple_net.py` | Basic CNN architecture from scratch |
| `simple_net_final.py` | Improved CNN with batch normalization and dropout |
| `my_resnet.py` | Custom ResNet18 implementation with transfer learning |
| `multilabel_resnet.py` | Multi-label classification for scene attributes |
| `data_transforms.py` | Data augmentation pipeline |
| `confusion_matrix.py` | Evaluation metrics and visualization |
| `optimizer.py` | Custom optimizer configurations |

## Models

### SimpleNet
A lightweight CNN built from scratch:
- 3 convolutional layers with ReLU and max pooling
- Batch normalization for training stability
- Dropout for regularization
- Fully connected classifier

### ResNet18 (Transfer Learning)
Fine-tuned ResNet18 pretrained on ImageNet:
- Frozen early layers, trainable later layers
- Modified final FC layer for 15-class output
- Achieves significantly higher accuracy than SimpleNet

### Multi-label ResNet
Extended architecture for predicting multiple scene attributes simultaneously (indoor/outdoor, natural/man-made, etc.)

## Trained Models

Pre-trained weights are included in `src/vision/`:
- `trained_SimpleNet_final.pt`
- `trained_SimpleNetFinal_final.pt`
- `trained_MyResNet18_final.pt`
- `trained_MultilabelResNet18_final.pt`

## Project Structure

```
├── src/vision/          # Model implementations and trained weights
├── assets/              # Dataset visualizations
├── docs/                # Project report
├── tests/               # Unit tests
└── proj3.ipynb          # Training and evaluation notebook
```

## Setup

```bash
conda env create -f conda/environment.yml
conda activate cv_proj3
pip install -e .
```

## Dataset

The project uses the 15-scene dataset. Scene attribute CSVs are included (`scene_attributes_train.csv`, `scene_attributes_test.csv`). Full image dataset can be downloaded separately.

## Usage

```python
from vision.my_resnet import MyResNet18
from vision.data_transforms import get_fundamental_transforms

# Load pretrained model
model = MyResNet18()
model.load_state_dict(torch.load('src/vision/trained_MyResNet18_final.pt'))
```
