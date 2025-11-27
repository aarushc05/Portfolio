import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    image_paths = []
    train_folder = os.path.join(dir_name, "train")
    test_folder = os.path.join(dir_name, "test")
    
    for folder in [train_folder, test_folder]:
        if os.path.exists(folder):
            class_folders = [d for d in os.listdir(folder) 
                           if os.path.isdir(os.path.join(folder, d))]
            for class_folder in class_folders:
                class_path = os.path.join(folder, class_folder)
                image_files = glob.glob(os.path.join(class_path, "*.jpg"))
                image_paths.extend(image_files)
    
    pixel_values = []
    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img) / 255.0
        pixel_values.extend(img_array.flatten())
    
    pixel_values = np.array(pixel_values)
    mean = np.mean(pixel_values)
    variance = np.var(pixel_values, ddof=1)
    std = np.sqrt(variance)

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
