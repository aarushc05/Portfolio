# SIFT Local Feature Matching & Camera Calibration

> Georgia Tech CS-6476 Computer Vision

## Overview

This project implements a complete local feature matching pipeline from scratch, including Harris corner detection, SIFT-like feature descriptors, and camera geometry estimation. The system can match corresponding points between images of the same scene taken from different viewpoints.

## Key Implementations

| File | Description |
|------|-------------|
| `part1_harris_corner.py` | Harris corner detector with non-maximum suppression |
| `part2_feature_matching.py` | Feature matching using ratio test |
| `part3_sift_descriptor.py` | SIFT-like gradient histogram descriptors |
| `part4_projection_matrix.py` | Camera projection matrix estimation |
| `part5_fundamental_matrix.py` | Fundamental matrix for epipolar geometry |
| `part6_ransac.py` | RANSAC for robust estimation with outliers |

## Algorithms Implemented

- **Harris Corner Detection**: Second moment matrix, corner response function, non-maximum suppression
- **SIFT Descriptors**: Gradient magnitude/orientation, 4×4 spatial bins, 8 orientation bins (128-dim vector)
- **Feature Matching**: Nearest neighbor with Lowe's ratio test for ambiguity rejection
- **Camera Calibration**: Direct Linear Transform (DLT) for projection matrix
- **Epipolar Geometry**: 8-point algorithm for fundamental matrix estimation
- **RANSAC**: Random sample consensus for robust fitting with outliers

## Results

Sample feature matching results are in `results/`:
- `vis_circles.jpg` — Detected keypoints
- `vis_lines.jpg` — Matched feature correspondences
- `eval.jpg` — Evaluation visualization

## Project Structure

```
├── src/vision/          # Implementation files
├── data/                # Test image pairs (Notre Dame, Mt. Rushmore, Gaudi)
├── results/             # Output visualizations
├── docs/                # Project report
├── tests/               # Unit tests
└── project-2.ipynb      # Main notebook
```

## Setup

```bash
conda env create -f conda/environment.yml
conda activate cv_proj2
pip install -e .
```

## Usage

Run the Jupyter notebook `project-2.ipynb` to see the full pipeline in action, or import individual modules:

```python
from vision.part1_harris_corner import compute_harris_response
from vision.part3_sift_descriptor import get_SIFT_descriptors
from vision.part6_ransac import ransac_fundamental_matrix
```
