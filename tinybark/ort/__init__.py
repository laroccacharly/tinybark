"""Bark on ONNX Runtime, from graphs exported by `export.py` and cached in
~/.cache/tinybark: the encoder on the GPU through WebGPU (Vulkan), the decoder
and joint on the CPU, where their small per-step calls are fastest. Any audio
length runs on the same graphs: no warm-up per length.

    bark = tinybark.ort.load()
    text = bark.transcribe(samples)  # 16 kHz mono float32 NumPy array
"""

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .. import audio, weights
from ..text import Tokenizer

# Bump when export.py changes, so stale graphs aren't loaded.
FORMAT = 2
# Token <|endoftext|>, generation_config.json's eos_token_id: generate() stops on it.
EOS = 3


def subsampled(frames: int) -> int:
    """Length after the encoder's three stride-2 convolutions (kernel 3, padding 1)."""
    for _ in range(3):
        frames = (frames - 1) // 2 + 1
    return frames


def directory() -> Path:
    return weights.CACHE / f"onnx-{weights.REVISION[:12]}-v{FORMAT}"


class Bark:
    def __init__(self, path: Path):
        config = json.loads((path / "config.json").read_text())
        self.blank = config["blank_token_id"]
        self.durations = config["durations"]
        self.max_symbols = config["max_symbols_per_step"]
        self.state_shape = (config["num_decoder_layers"], 1, config["decoder_hidden_size"])
        options = ort.SessionOptions()
        # One thread per physical core: SMT siblings slow the matmuls down.
        options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        self.encoder = ort.InferenceSession(path / "encoder.onnx", options, providers=["WebGpuExecutionProvider", "CPUExecutionProvider"])
        self.decoder, self.joint = (ort.InferenceSession(path / f"{name}.onnx", options, providers=["CPUExecutionProvider"]) for name in ("decoder", "joint"))
        self.tokenizer = Tokenizer(weights.tokenizer_file())

    def tokens(self, samples: np.ndarray) -> list[int]:
        """Greedy TDT decoding of 16 kHz mono float32 samples, as ParakeetForTDT.generate()."""
        feats = audio.features(samples)
        frames = len(feats)
        if frames < 2:
            return []
        (encoded,) = self.encoder.run(None, {"features": feats[None]})
        h = c = np.zeros(self.state_shape, np.float32)

        def predict(token: int):
            nonlocal h, c
            decoded, h, c = self.decoder.run(None, {"token": np.array([[token]]), "h": h, "c": c})
            return decoded

        # generate() starts from the blank token, reuses the decoder's output while
        # it predicts blanks, and sizes its output by the extractor's frame count.
        decoded = predict(self.blank)
        out, frame = [], 0
        for _ in range(self.max_symbols * subsampled(frames + 1) - 1):
            if frame >= len(encoded):
                break
            token, duration = self.joint.run(None, {"encoded": encoded[frame], "decoded": decoded})
            token, duration = int(token), self.durations[int(duration)]
            if token == self.blank:
                frame += max(duration, 1)
                continue
            out.append(token)
            if token == EOS:
                break
            decoded = predict(token)
            frame += duration
        return out

    def transcribe(self, samples: np.ndarray) -> str:
        return self.tokenizer.decode(self.tokens(samples))

    def transcribe_pcm(self, pcm: bytes) -> str:
        """16 kHz s16le mono PCM, as stt records it."""
        return self.transcribe(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768)


def prepare() -> Path:
    """Exports the graphs on first use (about a minute), which needs the `export` extra (torch)."""
    path = directory()
    if not (path / "config.json").exists():
        from . import export

        export.export(path)
    return path


def load() -> Bark:
    return Bark(prepare())
