"""
Data Loading and Preprocessing Utilities

Utility functions for loading and preprocessing various datasets
used in machine learning experiments.

Author: Aarush Chhiber
"""

import os

import cv2
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def get_housing_dataset():
    """
    Load and preprocess the California housing dataset.
    
    Converts the regression targets to 3 classes (low, medium, high)
    for multi-class classification experiments.
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test) as numpy arrays
               Labels are one-hot encoded
    """
    dataset = fetch_california_housing()
    x, y = dataset.data, dataset.target
    y = y.reshape(-1, 1)
    
    # Subsample for faster experimentation
    perm = np.random.RandomState(seed=3).permutation(x.shape[0])[:500]
    x = x[perm]
    y = y[perm]

    # Sort by target value for consistent class boundaries
    index_array = np.argsort(y.flatten())
    x, y = x[index_array], y[index_array]

    # Convert to 3 classes: low, medium, high
    values_per_list = len(y) // 3
    list1 = y[:values_per_list]
    list2 = y[values_per_list : 2 * values_per_list]
    list3 = y[2 * values_per_list :]
    
    label_mapping = {
        tuple(value): label
        for label, value_list in enumerate([list1, list2, list3])
        for value in value_list
    }
    updated_values = [label_mapping[tuple(value)] for value in y]
    
    # One-hot encode labels
    num_classes = len(set(updated_values))
    one_hot_encoded = np.eye(num_classes)[updated_values]
    y = np.array(one_hot_encoded)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    # Normalize features
    x_scale = MinMaxScaler()
    x_train = x_scale.fit_transform(x_train)
    x_test = x_scale.transform(x_test)

    return x_train, y_train, x_test, y_test


def get_mri_dataset(classes):
    """
    Load and preprocess the brain tumor MRI dataset.
    
    Handles caching of preprocessed data to speed up subsequent loads.
    Applies bilateral filtering and resizing to all images.
    
    Args:
        classes (list): List of class names to load
        
    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    save_dir = "./data/brain-tumor/"
    save_paths = {
        "x_train": os.path.join(save_dir, "x_train.pt"),
        "y_train": os.path.join(save_dir, "y_train.pt"),
        "x_test": os.path.join(save_dir, "x_test.pt"),
        "y_test": os.path.join(save_dir, "y_test.pt"),
    }

    # Load cached data if available
    if all(os.path.exists(path) for path in save_paths.values()):
        print("Loading preprocessed data from disk...")
        x_train = torch.load(save_paths["x_train"])
        y_train = torch.load(save_paths["y_train"])
        x_test = torch.load(save_paths["x_test"])
        y_test = torch.load(save_paths["y_test"])
        return x_train, y_train, x_test, y_test

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    image_size = 84

    for label in classes:
        # Load training images
        path = "./data/brain-tumor/Training"
        trainPath = os.path.join(path, label)
        for file in tqdm(
            os.listdir(trainPath),
            desc=f"Loading {label} training samples",
        ):
            image = cv2.imread(os.path.join(trainPath, file), 0)
            image = crop_margins(image)
            image = cv2.bilateralFilter(image, 2, 50, 50)
            image = cv2.resize(image, (image_size, image_size))
            x_train.append(image)
            y_train.append(classes.index(label))

        # Load test images
        path = "./data/brain-tumor/Testing"
        testPath = os.path.join(path, label)
        for file in tqdm(
            os.listdir(testPath),
            desc=f"Loading {label} test samples",
        ):
            image = cv2.imread(os.path.join(testPath, file), 0)
            image = crop_margins(image)
            image = cv2.bilateralFilter(image, 2, 50, 50)
            image = cv2.resize(image, (image_size, image_size))
            x_test.append(image)
            y_test.append(classes.index(label))

    # Cache preprocessed data
    torch.save(x_train, save_paths["x_train"])
    torch.save(y_train, save_paths["y_train"])
    torch.save(x_test, save_paths["x_test"])
    torch.save(y_test, save_paths["y_test"])
    print("Preprocessed data saved.")

    return x_train, y_train, x_test, y_test


def crop_margins(image):
    """
    Crop left and right margins from MRI images.
    
    Removes 60 pixels from each side to focus on the brain region.
    
    Args:
        image (np.ndarray): Input grayscale image
        
    Returns:
        np.ndarray: Cropped image
    """
    _, width = image.shape[:2]
    left_margin = 60
    right_margin = width - left_margin
    cropped_image = image[:, left_margin:right_margin]
    return cropped_image


def clean_text(text):
    """
    Clean and normalize text for character-level modeling.
    
    Applies the following transformations:
        - Convert to lowercase
        - Normalize whitespace
        - Normalize line endings
        - Keep only allowed characters
        - Fix punctuation spacing
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned text
    """
    text = text.lower()
    text = " ".join(text.split())
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    
    # Keep only allowed characters
    allowed_chars = set("abcdefghijklmnopqrstuvwxyz0123456789.,!?'\"\n ;:-")
    text = "".join(c for c in text if c in allowed_chars)
    
    # Fix punctuation
    for punct in ".,!?":
        text = text.replace(punct + punct, punct)
    for punct in ".,!?;:":
        text = text.replace(punct + " ", punct)
        text = text.replace(punct, punct + " ")
    for punct in ".,!?;:":
        text = text.replace(" " + punct, punct)
    text = text.replace('" ', '"').replace(' "', '"')
    text = text.replace("' ", "'").replace(" '", "'")
    text = " ".join(text.split())

    return text


def preprocess_text_data(text):
    """
    Preprocess text data for character-level sequence modeling.
    
    Creates input-output pairs where each input is a sequence
    of characters and the output is the next character.
    
    Args:
        text (str): Raw text to process
        
    Returns:
        dict: Contains:
            - x: Input sequences as indices
            - y: Target character indices
            - text: Cleaned text
            - char_indices: Char to index mapping
            - indices_char: Index to char mapping
            - vocab: List of unique characters
            - vocab_size: Number of unique characters
            - sequence_len: Length of input sequences
    """
    sequence_len = 30
    sliding_window_step = 1

    text = clean_text(text)

    vocab = sorted(set(text))
    vocab_size = len(vocab)

    # Create character mappings
    char_indices = {c: i for i, c in enumerate(vocab)}
    indices_char = {i: c for i, c in enumerate(vocab)}

    # Create sequences
    sentences = []
    next_chars = []
    for i in range(0, len(text) - sequence_len, sliding_window_step):
        sentences.append(text[i : i + sequence_len])
        next_chars.append(text[i + sequence_len])

    # Convert to arrays
    x = np.zeros((len(sentences), sequence_len), dtype=np.float32)
    y = np.zeros((len(sentences), 1), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t] = char_indices[char]
        y[i] = char_indices[next_chars[i]]

    return {
        "x": x,
        "y": y,
        "text": text,
        "char_indices": char_indices,
        "indices_char": indices_char,
        "vocab": vocab,
        "vocab_size": vocab_size,
        "sequence_len": sequence_len,
    }
