"""Bark in transformers' ParakeetForTDT on PyTorch, in fp16 on the GPU through ROCm
or in bf16 on the CPU, optionally with the encoder compiled by torch.compile.

    bark = tinybark.pytorch.load("gpu", compile=True)
    text = bark.transcribe(samples)  # 16 kHz mono float32 NumPy array

Needs the `pytorch` extra, with torch from the ROCm index.
"""

import os

# PyTorch's ROCm wheels ship no rocBLAS kernels for gfx1103 (Radeon 760M/780M):
# running as gfx1100, the same RDNA3 generation, works. Must be set before torch loads HIP.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import numpy as np
import torch
from transformers import AutoProcessor

from .. import weights
from .model import torch_model


class Bark:
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16, compile: bool = False):
        self.target, self.dtype = torch.device(device), dtype
        self.device = torch.cuda.get_device_name(self.target) if self.target.type == "cuda" else device
        self.model = torch_model(dtype).to(self.target)
        if compile:
            # generate() runs the encoder once per call and the decoder and joint once
            # per step: the encoder is where the work is.
            self.model.encoder = torch.compile(self.model.encoder)
        self.processor = AutoProcessor.from_pretrained(weights.TOKENIZER_REPO)
        self.skip = {self.model.config.blank_token_id, self.model.config.pad_token_id}
        self.tokenizer = self.processor.tokenizer

    def tokens(self, samples: np.ndarray) -> list[int]:
        inputs = self.processor([samples], sampling_rate=16_000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.target, self.dtype if v.dtype.is_floating_point else v.dtype) for k, v in inputs.items()}
        with torch.inference_mode():
            sequence = self.model.generate(**inputs).sequences[0].tolist()
        # The sequence starts with the decoder's start token.
        return [t for t in sequence[1:] if t not in self.skip]

    def transcribe(self, samples: np.ndarray) -> str:
        return self.processor.batch_decode([self.tokens(samples)], skip_special_tokens=True)[0].strip()

    def transcribe_pcm(self, pcm: bytes) -> str:
        """16 kHz s16le mono PCM, as stt records it."""
        return self.transcribe(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768)


def load(device: str = "gpu", compile: bool = False) -> Bark:
    """On the "gpu" in fp16, or the "cpu" in bf16 (fp32 without AVX512 BF16), as stt's bark.py runs it."""
    if device == "gpu":
        return Bark("cuda", torch.float16, compile)
    bf16 = getattr(torch.cpu, "_is_avx512_bf16_supported", lambda: False)()
    return Bark("cpu", torch.bfloat16 if bf16 else torch.float32, compile)
