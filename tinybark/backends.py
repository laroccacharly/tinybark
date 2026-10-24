"""Every backend behind one interface: prepare() once (downloads, exports, builds;
no GPU), then load() a model with tokens(samples) and transcribe(samples).

    bark = tinybark.backends.BACKENDS["onnx"].load()

Each backend's `env` must be set before the process imports it: run one backend
per process."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from . import weights


class Model(Protocol):
    device: str

    def tokens(self, samples: np.ndarray) -> list[int]: ...

    def transcribe(self, samples: np.ndarray) -> str: ...


@dataclass(frozen=True)
class Backend:
    name: str
    description: str
    prepare: Callable[[], object]
    load: Callable[[], Model]
    env: dict[str, str] = field(default_factory=dict)


def _pytorch(compile: bool) -> Model:
    from . import pytorch

    return pytorch.load(compile=compile)


def _prepare_pytorch() -> None:
    from transformers import AutoProcessor

    weights.decoded()
    AutoProcessor.from_pretrained(weights.TOKENIZER_REPO)


def _tinygrad() -> Model:
    from tinygrad import Device

    from . import tinygrad

    # Bark() without load()'s warm-up: the first call at each padded length is cold.
    bark = tinygrad.Bark()
    bark.device = Device.DEFAULT
    return bark


def _prepare_tinygrad() -> None:
    weights.converted()
    weights.tokenizer_file()


def _onnx() -> Model:
    from . import ort

    bark = ort.load()
    bark.device = "encoder " + bark.encoder.get_providers()[0] + ", decoder and joint CPU"
    return bark


def _prepare_onnx() -> None:
    from . import ort

    ort.prepare()
    weights.tokenizer_file()


def _mojo() -> Model:
    from . import mojo

    return mojo.load()


def _prepare_mojo() -> None:
    from . import mojo

    mojo.build()


BACKENDS = {
    b.name: b
    for b in [
        Backend("pytorch", "transformers on PyTorch, fp16, ROCm GPU", _prepare_pytorch, lambda: _pytorch(compile=False)),
        Backend("pytorch-compile", "transformers on PyTorch, fp16, ROCm GPU, encoder through torch.compile", _prepare_pytorch, lambda: _pytorch(compile=True)),
        Backend("tinygrad", "tinygrad, fp16, AMD GPU (no warm-up)", _prepare_tinygrad, _tinygrad, {"DEV": "AMD"}),
        Backend("onnx", "ONNX Runtime, encoder fp16 on WebGPU, decoder and joint on the CPU", _prepare_onnx, _onnx),
        Backend("mojo", "Mojo, encoder fp16 on the GPU through HIP, decoder and joint on the CPU", _prepare_mojo, _mojo),
    ]
}
