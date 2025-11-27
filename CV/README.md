# Computer Vision

> Georgia Tech CS-6476: Graduate Computer Vision

This directory contains projects from Georgia Tech's graduate-level Computer Vision course. Each project implements fundamental CV algorithms from scratch, progressing from classical techniques to deep learning approaches.

---

## Projects

### [sift-feature-matching](./sift-feature-matching/)
Implements a complete local feature matching pipeline including Harris corner detection, SIFT descriptors, and camera geometry estimation (fundamental matrix, RANSAC). Matches corresponding points between images taken from different viewpoints.

### [scene-recognition-cnn](./scene-recognition-cnn/)
Scene classification using convolutional neural networks. Progresses from a custom CNN architecture to transfer learning with ResNet18, classifying images into 15 scene categories with multi-label attribute prediction.

### [semantic-segmentation](./semantic-segmentation/)
Pixel-wise semantic segmentation using PSPNet (Pyramid Scene Parsing Network). Implements the Pyramid Pooling Module for multi-scale context aggregation on driving scene datasets (CamVid, KITTI).

### [pointnet-3d](./pointnet-3d/)
3D point cloud classification using PointNet. Processes raw (x, y, z) coordinates directly without voxelization, achieving permutation invariance through symmetric functions. Includes T-Net spatial transformer for input alignment.

