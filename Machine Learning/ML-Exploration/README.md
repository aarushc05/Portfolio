# Machine Learning From Scratch

A deep dive into machine learning fundamentals through hands-on implementations. This project builds core ML algorithms from the ground up to develop intuition for how they actually work under the hood.

**Author:** Aarush Chhiber

---

## Purpose

The goal of this project was to move beyond just *using* machine learning libraries and actually *understand* the mathematics and mechanics that power them. By implementing algorithms from scratch, I gained insight into:

- How gradients flow backward through a neural network
- Why certain activation functions work better than others
- How dropout prevents overfitting at a fundamental level
- The difference between vanilla SGD and adaptive optimizers like Adam
- How CNNs extract hierarchical features from images
- Why LSTMs can capture long-range dependencies that simple RNNs cannot

---

## Datasets

| Dataset | Source | Task | Size |
|---------|--------|------|------|
| **California Housing** | sklearn | Regression → Classification | 500 samples, 8 features |
| **Brain Tumor MRI** | Kaggle | Medical Image Classification | 7,023 images, 4 classes |
| **Dry Bean** | UCI ML Repository | Multi-class Classification | 13,611 samples, 16 features |
| **Shakespeare Text** | TensorFlow Datasets | Character-level Generation | 50,000 characters |

---

## Models Implemented

### 1. Neural Network from Scratch (NumPy)

A two-layer fully connected network built entirely with NumPy—no TensorFlow, no PyTorch.

**Architecture:**
```
Input (8) → Dense (15) → SiLU → Dropout → Dense (7) → SiLU → Dense (3) → Softmax
```

**What I implemented:**
- **Forward propagation** - Matrix multiplications and activation functions
- **Backpropagation** - Computing gradients via the chain rule
- **SiLU activation** - A smooth, self-gated activation function
- **Dropout** - Inverted dropout with proper scaling
- **Cross-entropy loss** - For multi-class classification
- **SGD & Adam optimizers** - Including momentum and adaptive learning rates
- **Mini-batch gradient descent** - For efficient training

**Key insight:** Implementing backprop manually made me truly understand why we cache forward pass values and how gradients flow backward layer by layer.

---

### 2. Convolutional Neural Network (PyTorch)

A CNN for classifying brain MRI scans into 4 categories: glioma, meningioma, no tumor, and pituitary tumors.

**Architecture:**
```
Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → AdaptiveAvgPool → FC → FC
```

**What I implemented:**
- **Data augmentation** - Random rotation, flipping, and translation
- **Custom training loop** - With learning rate scheduling
- **Metrics tracking** - Loss and accuracy per epoch

**Key insight:** Data augmentation was crucial for this medical imaging task. Without it, the model quickly overfit to the small training set.

---

### 3. Random Forest Classifier

An ensemble learning approach using bootstrap aggregating (bagging) with ExtraTreeClassifiers.

**What I implemented:**
- **Bootstrapping** - Random sampling with replacement for rows, without replacement for features
- **Out-of-bag scoring** - Using samples not in each tree's bootstrap for validation
- **Hyperparameter grid search** - Automated tuning of n_estimators, max_depth, max_features
- **AdaBoost variant** - Adaptive boosting with weighted voting
- **Feature importance** - Visualizing which features matter most

**Key insight:** The OOB score is a clever way to get validation metrics without needing a separate validation set—each sample is "out of bag" for about 37% of trees.

---

### 4. RNN & LSTM for Text Generation (TensorFlow)

Character-level language models that learn to generate Shakespeare-like text.

**Architectures:**
```
Simple RNN: Embedding → SimpleRNN → Dense → Softmax
LSTM:       Embedding → LSTM → Dense → Softmax
```

**What I implemented:**
- **Character-level tokenization** - Mapping characters to indices
- **Sequence generation** - Sliding window approach for training data
- **Temperature sampling** - Controlling randomness in generation
- **Model checkpointing** - Saving best weights during training

**Key insight:** The LSTM consistently produced more coherent text than the simple RNN, demonstrating how gating mechanisms help with longer-range dependencies.

---

## Results

| Model | Task | Performance |
|-------|------|-------------|
| Neural Network (Adam) | Housing Price Classification | ~68% accuracy |
| CNN | Brain Tumor Classification | >80% accuracy |
| Random Forest | Dry Bean Classification | >85% accuracy |
| LSTM | Text Generation | Coherent Shakespeare-style output |

---

## Setup

### Quick Start

```bash
# Create environment
cd environment
conda env create -f environment.yml
conda activate ml_portfolio

# Register Jupyter kernel
python -m ipykernel install --user --name ml_portfolio --display-name "ML Portfolio"

# Run the notebook
jupyter notebook ../ml_portfolio.ipynb
```

### Requirements

- Python 3.10
- TensorFlow 2.15
- PyTorch 2.x
- NumPy, pandas, matplotlib, scikit-learn

See `environment/environment.yml` for the complete specification.

---

## Project Structure

```
├── ml_portfolio.ipynb          # Main notebook with all experiments
├── NN.py                       # Neural network from scratch
├── cnn.py                      # CNN architecture
├── cnn_trainer.py              # Training framework
├── cnn_image_transformations.py # Data augmentation
├── random_forest.py            # Random forest implementation
├── rnn.py                      # Simple RNN model
├── lstm.py                     # LSTM model
├── base_sequential_model.py    # Base class for RNN/LSTM
├── text_generator.py           # Text generation utilities
├── utilities/
│   └── utils.py                # Data loading helpers
├── data/
│   ├── brain-tumor/            # MRI images (preprocessed)
│   ├── Dry_Bean_Dataset.csv    # Bean classification data
│   └── images/                 # Architecture diagrams
└── environment/
    ├── environment.yml         # Conda environment
    ├── requirements.txt        # Pip requirements
    └── environment_setup.md    # Setup instructions
```

---

## Key Takeaways

1. **Implementing backprop manually** is the best way to understand neural networks. Debugging gradient shapes forces you to really understand the math.

2. **Adam optimizer** converges faster and more reliably than vanilla SGD, but understanding SGD first makes Adam's improvements clear.

3. **Dropout** is simple but powerful—randomly zeroing neurons during training creates an implicit ensemble effect.

4. **Data augmentation** can be more important than model architecture, especially for small datasets.

5. **Ensemble methods** like Random Forests are remarkably robust and often competitive with neural networks on tabular data.

6. **LSTMs vs RNNs** - The difference in text generation quality really demonstrates why gating mechanisms matter for sequential data.

---

## References

- [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- [Adam Optimizer Paper](https://arxiv.org/abs/1412.6980)
- [Dropout Paper](https://jmlr.org/papers/v15/srivastava14a.html)

