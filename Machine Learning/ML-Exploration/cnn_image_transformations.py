"""
Image Transformations for CNN Training

Data augmentation and preprocessing utilities for medical image classification.
Provides training and testing transformations using torchvision.

Author: Aarush Chhiber
"""

import torch
from PIL import Image
from torchvision.transforms import v2


def create_training_transformations():
    """
    Create data augmentation pipeline for training.
    
    Applies the following transformations:
        1. Convert to tensor image
        2. Random rotation (±10 degrees)
        3. Random translation (±5% in x and y)
        4. Random horizontal flip (50% probability)
        5. Convert to float32 with normalization
    
    Data augmentation helps prevent overfitting by creating
    variations of the training images.
    
    Returns:
        v2.Compose: Composed transformation pipeline
    """
    return v2.Compose([
        v2.ToImage(),
        v2.RandomRotation(degrees=10),
        v2.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
    ])


def create_testing_transformations():
    """
    Create preprocessing pipeline for testing/inference.
    
    Applies minimal transformations to preserve image fidelity:
        1. Convert to tensor image
        2. Convert to float32 with normalization
    
    No augmentation is applied during testing to ensure
    consistent evaluation.
    
    Returns:
        v2.Compose: Composed transformation pipeline
    """
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


class TransformedDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset with preprocessing transformations.
    
    Stores pre-transformed images in memory for efficient training.
    Handles conversion from numpy arrays to PIL images before
    applying transformations.
    
    Attributes:
        data (torch.Tensor): Preprocessed image tensors
        targets (torch.Tensor): Class labels
        transform: Transformation pipeline to apply
    """
    
    def __init__(self, images, labels, transform=None):
        """
        Initialize the transformed dataset.
        
        Args:
            images: List or array of images (numpy arrays)
            labels: List or array of integer labels
            transform: Transformation pipeline to apply
        """
        self.transform = transform
        self.data = []
        self.targets = torch.tensor(labels, dtype=torch.long)

        # Apply transformations and store
        for img in images:
            img = Image.fromarray(img.astype("uint8"))
            if self.transform:
                img = self.transform(img)
            self.data.append(img)

        # Stack into single tensor for efficient indexing
        self.data = torch.stack(self.data)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image tensor, label)
        """
        return self.data[idx], self.targets[idx]
