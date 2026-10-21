"""Underdog Bark 3.0 Clean speech-to-text on tinygrad.

    bark = tinybark.load()
    text = bark.transcribe(samples)  # 16 kHz mono float32 NumPy array

load() warms up every padded length up to 30 s in memory, about 5.6 s per 2.56 s
length: start it once with the server and keep it.
"""

import numpy as np

from . import weights
from .model import Parakeet
from .text import Tokenizer

__all__ = ["Bark", "load"]


class Bark:
    def __init__(self):
        tensors, config = weights.load()
        self.model = Parakeet(tensors, config)
        self.tokenizer = Tokenizer(weights.tokenizer_file())

    def tokens(self, samples: np.ndarray) -> list[int]:
        return self.model.generate(samples)

    def transcribe(self, samples: np.ndarray) -> str:
        return self.tokenizer.decode(self.tokens(samples))

    def transcribe_pcm(self, pcm: bytes) -> str:
        """16 kHz s16le mono PCM, as stt records it."""
        return self.transcribe(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768)


# Covers dictation up to 30 s.
MAX_SECONDS = 30.0


def load(max_seconds: float = MAX_SECONDS) -> Bark:
    """A warm model: audio up to `max_seconds` transcribes at full speed from the
    first call. Longer audio works too, but its first call per padded length is slow."""
    bark = Bark()
    bark.model.warm_up(round(max_seconds * 100))
    return bark
