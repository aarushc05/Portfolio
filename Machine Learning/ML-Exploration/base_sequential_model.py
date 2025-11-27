"""
Base Sequential Model for Text Generation

Abstract base class providing common functionality for sequential models
including training, weight saving, and loss visualization.

Author: Aarush Chhiber
"""

import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf


class BaseSequentialModel:
    """
    Abstract base class for sequential text generation models.
    
    Provides foundational structure for building and training
    sequential models with TensorFlow/Keras, including:
        - Model weight persistence
        - Training history tracking
        - Learning rate scheduling
        - Loss visualization
    
    Attributes:
        vocab_size (int): Size of the character vocabulary
        max_input_len (int): Maximum sequence length
        model (tf.keras.Model): The Keras model instance
        model_name (str): Name identifier for the model
        loss_history (dict): Training loss history
        hp (dict): Hyperparameters dictionary
    """

    def __init__(self, vocab_size, max_input_len):
        """
        Initialize the base sequential model.
        
        Args:
            vocab_size (int): Number of unique tokens in vocabulary
            max_input_len (int): Maximum length of input sequences
        """
        self.vocab_size = vocab_size
        self.max_input_len = max_input_len
        self.model = None
        self.model_name = ""
        self.loss_history = None
        self.hyper_params = {}

    def save_model_path(self):
        """
        Get the file path for saving model weights.
        
        Returns:
            str: Path in format 'rnn_model_weights/{model_name}_weights.keras'
        """
        return f"rnn_model_weights/{self.model_name}_weights.keras"

    def save_losses_path(self):
        """
        Get the file path for saving loss history.
        
        Returns:
            str: Path in format 'rnn_model_weights/{model_name}_losses.json'
        """
        return f"rnn_model_weights/{self.model_name}_losses.json"

    def get_callbacks(self):
        """
        Create training callbacks.
        
        Includes:
            - ModelCheckpoint: Saves best model weights
            - ReduceLROnPlateau: Reduces learning rate when loss plateaus
        
        Returns:
            list: List of Keras callbacks
        """
        os.makedirs("rnn_model_weights", exist_ok=True)
        save_model_path = self.save_model_path()
        
        return [
            tf.keras.callbacks.ModelCheckpoint(
                save_model_path, monitor="loss", save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.2, patience=1, min_lr=0.001
            ),
        ]

    def train(self, x, y, train_from_scratch=False):
        """
        Train the sequential model.
        
        Attempts to load existing weights if available. Falls back
        to training from scratch if loading fails or if specified.
        
        Args:
            x (array-like): Input sequences
            y (array-like): Target values
            train_from_scratch (bool): Force training from scratch
        """
        save_model_path = self.save_model_path()
        save_losses_path = self.save_losses_path()

        if train_from_scratch is False and os.path.exists(save_model_path):
            try:
                self.model.load_weights(save_model_path)
                if os.path.exists(save_losses_path):
                    with open(save_losses_path, "r") as f:
                        self.loss_history = json.load(f)
                print(f"Loaded saved {self.model_name} model and weights.")
            except Exception:
                print(
                    "Could not load pre-trained model (possible architecture mismatch). "
                    "Training from scratch..."
                )
                self.train(x, y, True)
        else:
            print(f"Training {self.model_name} model from scratch...")
            
            batch_size = self.hp["batch_size"]
            epochs = self.hp["epochs"]
            model_callbacks = self.get_callbacks()
            
            # Convert targets to one-hot encoding
            y_onehot = tf.one_hot(y, depth=self.vocab_size)
            y_onehot = tf.squeeze(y_onehot, axis=1)
            
            full_history = self.model.fit(
                x,
                y_onehot,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=model_callbacks,
                verbose=1,
            )
            
            print(f"Saved {self.model_name} model weights to {save_model_path}")

            # Save loss history
            self.loss_history = {"losses": full_history.history["loss"]}
            with open(save_losses_path, "w") as f:
                json.dump(self.loss_history, f)
            print(f"Saved {self.model_name} loss history to {save_losses_path}")

    def plot_loss(self):
        """
        Plot the training loss history.
        
        Displays a line plot of loss values across training epochs.
        """
        if self.loss_history is None:
            print("No training history available. Train the model first.")
            return

        losses = self.loss_history["losses"]
        plt.figure(figsize=(8, 5))
        plt.plot(losses, "b-")
        plt.title(f"{self.model_name} Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(range(len(losses)))
        plt.tight_layout()
        plt.show()
