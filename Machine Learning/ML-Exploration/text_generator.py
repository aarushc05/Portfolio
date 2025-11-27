"""
Text Generator for Character-Level Models

Utility class for generating text using trained RNN/LSTM models
with temperature-controlled sampling.

Author: Aarush Chhiber
"""

import numpy as np


class TextGenerator:
    """
    Text generation utility for character-level language models.
    
    Uses temperature sampling to control the randomness of generated
    text. Higher temperatures produce more diverse but potentially
    less coherent text, while lower temperatures produce more
    conservative, predictable text.
    
    Attributes:
        char_indices (dict): Mapping from characters to indices
        indices_char (dict): Mapping from indices to characters
        max_input_len (int): Maximum sequence length for the model
    """

    def __init__(self, char_indices, indices_char, max_input_len):
        """
        Initialize the text generator.
        
        Args:
            char_indices (dict): Character to index mapping
            indices_char (dict): Index to character mapping
            max_input_len (int): Maximum input sequence length
        """
        self.char_indices = char_indices
        self.indices_char = indices_char
        self.max_input_len = max_input_len

    def sample(self, preds, temperature=0.5):
        """
        Sample next character index using temperature scaling.
        
        Temperature controls the randomness of predictions:
            - Low temperature (< 1): More deterministic, picks high prob chars
            - Temperature = 1: Original probability distribution
            - High temperature (> 1): More random, uniform distribution
        
        Args:
            preds (array-like): Probability distribution over vocabulary
            temperature (float): Sampling temperature (default 0.5)
            
        Returns:
            int: Sampled character index
        """
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate(self, model, seed_text, length=150, temperature=0.5):
        """
        Generate text continuation from a seed string.
        
        Pads or truncates the seed text to match the model's
        expected input length, then generates characters one
        at a time by sampling from the model's predictions.
        
        Args:
            model: Trained language model with predict method
            seed_text (str): Initial text to continue from
            length (int): Number of characters to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated text (printed to console)
        """
        # Pad or truncate seed text
        seed_text = (
            " " * (self.max_input_len - len(seed_text)) + seed_text
            if len(seed_text) < self.max_input_len
            else seed_text[-self.max_input_len:]
        )

        generated = ""
        print(f"-------------------- {model.model_name} Model --------------------")
        print("Prompt: " + seed_text)
        print("Model: ", end="")

        for _ in range(length):
            # Encode seed text
            x_pred = np.zeros((1, self.max_input_len))
            for t, char in enumerate(seed_text):
                x_pred[0, t] = self.char_indices[char]

            # Get prediction and sample
            preds = model.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, temperature)
            next_char = self.indices_char[next_index]

            generated += next_char
            seed_text = seed_text[1:] + next_char
            print(next_char, end="")
        
        print()
        return generated
