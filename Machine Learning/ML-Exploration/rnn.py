"""
Simple RNN Model for Character-Level Text Generation

A vanilla Recurrent Neural Network implementation using TensorFlow/Keras
for next-character prediction in text sequences.

Author: Aarush Chhiber
"""

import tensorflow as tf
from base_sequential_model import BaseSequentialModel


class RNN(BaseSequentialModel):
    """
    Simple RNN model for character-level text generation.
    
    Uses a basic recurrent architecture to learn patterns in 
    sequential text data for character prediction.
    
    Architecture:
        Embedding -> SimpleRNN -> Dense -> Softmax
    
    Attributes:
        vocab_size (int): Size of the character vocabulary
        max_input_len (int): Maximum sequence length for input
        hp (dict): Hyperparameters for training
    """
    
    def __init__(self, vocab_size, max_input_len):
        """
        Initialize the RNN model.
        
        Args:
            vocab_size (int): Number of unique characters in vocabulary
            max_input_len (int): Maximum length of input sequences
        """
        super().__init__(vocab_size, max_input_len)
        self.model_name = "RNN"

    def set_hyperparameters(self):
        """
        Configure training hyperparameters.
        
        Sets embedding dimension, RNN units, learning rate,
        batch size, and number of training epochs.
        """
        self.hp = {
            "embedding_dim": 256,
            "rnn_units": 128,
            "learning_rate": 0.01,
            "batch_size": 128,
            "epochs": 10,
        }

    def define_model(self):
        """
        Build the Simple RNN model architecture.
        
        Creates a sequential model with:
            1. Embedding layer: Maps character indices to dense vectors
            2. SimpleRNN layer: Processes sequences with recurrent connections
            3. Dense layer: Maps RNN output to vocabulary size
            4. Softmax activation: Produces probability distribution
        """
        embedding_dim = self.hp["embedding_dim"]
        rnn_units = self.hp["rnn_units"]
        vocab_size = self.vocab_size
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
            tf.keras.layers.SimpleRNN(rnn_units),
            tf.keras.layers.Dense(vocab_size),
            tf.keras.layers.Activation('softmax')
        ])

    def build_model(self):
        """
        Compile and build the model.
        
        Uses RMSprop optimizer with categorical cross-entropy loss,
        which works well for recurrent models.
        """
        learning_rate = self.hp["learning_rate"]
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy")
        self.model.build((None, self.max_input_len))
