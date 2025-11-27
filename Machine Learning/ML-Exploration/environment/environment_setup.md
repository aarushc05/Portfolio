# Environment Setup

This project requires Python 3.10 with TensorFlow 2.15 and PyTorch.

---

## Option 1: Conda (Recommended)

### Create and activate the environment:

```bash
cd environment
conda env create -f environment.yml
conda activate ml_portfolio
```

### Register the kernel for Jupyter:

```bash
python -m ipykernel install --user --name ml_portfolio --display-name "ML Portfolio"
```

### Then open the notebook and select the "ML Portfolio" kernel.

---

## Option 2: Pip with Virtual Environment

### Create virtual environment:

```bash
python3.10 -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
```

### Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Register the kernel:

```bash
python -m ipykernel install --user --name ml_env --display-name "ML Portfolio"
```

---

## Troubleshooting

### TensorFlow Issues

If you get TensorFlow errors, try:

```bash
pip uninstall tensorflow keras
pip install tensorflow==2.15.0
```

### PyTorch with GPU (Optional)

For CUDA support, install PyTorch separately:

```bash
# Check https://pytorch.org/get-started/locally/ for your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Apple Silicon (M1/M2/M3)

TensorFlow works natively on Apple Silicon with the above setup. For optimized performance:

```bash
pip install tensorflow-metal
```

---

## Verify Installation

Run this in Python to verify everything is installed:

```python
import numpy as np
import tensorflow as tf
import torch
import sklearn

print(f"NumPy: {np.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"Keras (via TF): {tf.keras.__version__}")
print("✓ All packages installed correctly!")
```
