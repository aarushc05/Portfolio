"""
Convolutional Neural Network for Medical Image Classification

A PyTorch CNN implementation for brain tumor MRI classification.
Designed for 4-class classification: glioma, meningioma, no tumor, pituitary.

Author: Aarush Chhiber
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Convolutional Neural Network for brain tumor MRI classification.
    
    Architecture follows a standard pattern of:
        - Feature extraction: Conv -> ReLU -> MaxPool (repeated)
        - Adaptive pooling for consistent output size
        - Classification: Fully connected layers
    
    Designed to handle grayscale medical images and classify into 4 categories.
    
    Attributes:
        feature_extractor (nn.Sequential): Convolutional feature extraction layers
        avg_pooling (nn.AdaptiveAvgPool2d): Adaptive pooling for fixed output
        classifier (nn.Sequential): Fully connected classification layers
    """
    
    def __init__(self):
        """
        Initialize the CNN architecture.
        
        Creates a network with:
            - 2 convolutional blocks with ReLU and max pooling
            - Adaptive average pooling to 7x7
            - 2 fully connected layers for classification
        """
        super().__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling for consistent output size
        self.avg_pooling = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # 4 classes for MRI dataset
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Handles both grayscale (1-channel) and RGB (3-channel) inputs
        by automatically converting grayscale to 3-channel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
                              where C is 1 or 3
        
        Returns:
            torch.Tensor: Class logits of shape (N, 4)
        """
        # Convert grayscale to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        x = self.feature_extractor(x)
        x = self.avg_pooling(x)
        x = self.classifier(x)
        
        return x
