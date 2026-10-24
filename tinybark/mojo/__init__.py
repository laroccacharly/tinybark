"""Bark in Mojo: the encoder in fp16 on an AMD GPU through HIP, the decoder and
joint on the CPU. main.mojo is built once into a binary cached in ~/.cache/tinybark
(keyed by a hash of the sources and the Mojo version) and runs as a server process; Bark talks to it
over stdin and stdout.

    bark = tinybark.mojo.load()
    text = bark.transcribe(samples)  # 16 kHz mono float32 NumPy array

Needs the `mojo` extra and ROCm's HIP runtime.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

import numpy as np

from .. import weights
from ..text import Tokenizer

SOURCES = Path(__file__).parent


def binary() -> Path:
    files = sorted([SOURCES / "main.mojo", *(SOURCES / "bark").glob("*.mojo")])
    digest = hashlib.sha256(metadata.version("mojo").encode() + b"".join(f.read_bytes() for f in files)).hexdigest()[:12]
    return weights.CACHE / f"mojo-bark-{digest}"


def build() -> Path:
    """Builds the binary if the sources changed (about a minute), and makes the weights it reads."""
    weights.converted()
    weights.tokenizer_file()
    path = binary()
    if not path.exists():
        mojo = Path(sys.executable).parent / "mojo"
        tmp = path.with_suffix(".tmp")
        subprocess.run([mojo, "build", "-I", SOURCES, SOURCES / "main.mojo", "-o", tmp], check=True)
        tmp.rename(path)
    return path


class Bark:
    def __init__(self, path: Path):
        self.process = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.device = self._reply("ready")
        self.tokenizer = Tokenizer(weights.tokenizer_file())
        # The samples go through a file in shared memory: Mojo's input() reads lines only.
        self.samples = Path(tempfile.mkstemp(prefix="tinybark-", suffix=".f32", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)[1])

    def _reply(self, kind: str) -> str:
        for line in self.process.stdout:
            if line.startswith(kind + " "):
                return line[len(kind) + 1 :].strip()
        raise RuntimeError(f"the Mojo server exited with {self.process.wait()}")

    def tokens(self, samples: np.ndarray) -> list[int]:
        np.ascontiguousarray(samples, np.float32).tofile(self.samples)
        self.process.stdin.write(f"{self.samples}\n")
        self.process.stdin.flush()
        return json.loads(self._reply("tokens"))

    def transcribe(self, samples: np.ndarray) -> str:
        return self.tokenizer.decode(self.tokens(samples))

    def transcribe_pcm(self, pcm: bytes) -> str:
        """16 kHz s16le mono PCM, as stt records it."""
        return self.transcribe(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768)

    def close(self) -> None:
        self.process.stdin.close()
        self.process.wait()
        self.samples.unlink(missing_ok=True)


def load() -> Bark:
    """Builds the binary on first use (see build()), then starts the server."""
    return Bark(build())
