"""
LSTM Model for Character-Level Text Generation

A Long Short-Term Memory network implementation using TensorFlow/Keras
for next-character prediction in text sequences.

Author: Aarush Chhiber
"""

import tensorflow as tf
from base_sequential_model import BaseSequentialModel


class LSTM(BaseSequentialModel):
    """
    LSTM-based model for character-level text generation.
    
    Long Short-Term Memory networks are designed to learn long-term 
    dependencies in sequential data, making them well-suited for 
    text generation tasks.
    
    Architecture:
        Embedding -> LSTM -> Dense -> Softmax
    
    Attributes:
        vocab_size (int): Size of the character vocabulary
        max_input_len (int): Maximum sequence length for input
        hp (dict): Hyperparameters for training
    """
    
    def __init__(self, vocab_size, max_input_len):
        """
        Initialize the LSTM model.
        
        Args:
            vocab_size (int): Number of unique characters in vocabulary
            max_input_len (int): Maximum length of input sequences
        """
        super().__init__(vocab_size, max_input_len)
        self.model_name = "LSTM"

    def set_hyperparameters(self):
        """
        Configure training hyperparameters.
        
        Sets embedding dimension, LSTM units, learning rate,
        batch size, and number of training epochs.
        """
        self.hp = {
            "embedding_dim": 256,
            "lstm_units": 128,
            "learning_rate": 0.01,
            "batch_size": 128,
            "epochs": 10,
        }

    def define_model(self):
        """
        Build the LSTM model architecture.
        
        Creates a sequential model with:
            1. Embedding layer: Maps character indices to dense vectors
            2. LSTM layer: Processes sequences and captures temporal patterns
            3. Dense layer: Maps LSTM output to vocabulary size
            4. Softmax activation: Produces probability distribution
        """
        embedding_dim = self.hp["embedding_dim"]
        lstm_units = self.hp["lstm_units"]
        vocab_size = self.vocab_size
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
            tf.keras.layers.LSTM(lstm_units),
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
