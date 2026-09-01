import abc
import random

import numpy as np


class TextGenerator(abc.ABC):
    """Abstract class representing an LLM generating text by returning logits"""

    @abc.abstractmethod
    def next_token_logits(self, prompt: str) -> list[float]:
        """
        Generates next token's logits

        Args:
            prompt (str): Prompt given to the model.

        Returns:
            list[float]: Logits.
        """
        pass


class HardWaterMarker:
    def __init__(self, seed: int | None, vocabulary_size: int, ratio: float) -> None:
        self.seed = seed
        generator = random.Random(seed)
        shuffled_indices = [i for i in range(vocabulary_size)]
        generator.shuffle(shuffled_indices)
        green_list_size = int(vocabulary_size * ratio)
        self.green_list = shuffled_indices[:green_list_size]
        self.red_list = shuffled_indices[green_list_size:]

    def update_logits(self, logits: list[float]) -> list[float]:
        """
        Updates the logits probabilities by randomly selecting a subset

        Args:
            logits (list[float]): Logits predicted by the model.

        Returns:
            tuple[list[float], list[int], list[int]]: Updated logits, green list and red list.
        """
        new_logits = logits.copy()
        for i in self.red_list:
            new_logits[i] = 0
        np.softm
