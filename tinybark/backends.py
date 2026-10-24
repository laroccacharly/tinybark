"""Every backend behind one interface: prepare() once (downloads, exports, builds;
no GPU), then load(device) a model with tokens(samples) and transcribe(samples),
on the "gpu" or the "cpu".

    bark = tinybark.backends.BACKENDS["onnx"].load("cpu")

A backend's env(device) must be set before the process imports it: run one
backend per process."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import weights

DEVICES = ("gpu", "cpu")


class Model(Protocol):
    device: str

    def tokens(self, samples: np.ndarray) -> list[int]: ...

    def transcribe(self, samples: np.ndarray) -> str: ...


@dataclass(frozen=True)
class Backend:
    name: str
    # What runs where, for each device the backend supports.
    descriptions: dict[str, str]
    prepare: Callable[[], object]
    load: Callable[[str], Model]
    envs: dict[str, dict[str, str]] | None = None

    def env(self, device: str) -> dict[str, str]:
        return (self.envs or {}).get(device, {})


def _pytorch(device: str, compile: bool) -> Model:
    from . import pytorch

    return pytorch.load(device, compile=compile)


def _prepare_pytorch() -> None:
    from transformers import AutoProcessor

    weights.decoded()
    AutoProcessor.from_pretrained(weights.TOKENIZER_REPO)


def _tinygrad(device: str) -> Model:
    from tinygrad import Device

    from . import tinygrad

    # Bark() without load()'s warm-up: the first call at each padded length is cold.
    bark = tinygrad.Bark()
    bark.device = Device.DEFAULT
    return bark


def _prepare_tinygrad() -> None:
    weights.converted()
    weights.tokenizer_file()


def _onnx(device: str) -> Model:
    from . import ort

    bark = ort.load(device)
    bark.device = "encoder " + bark.encoder.get_providers()[0] + ", decoder and joint CPU"
    return bark


def _prepare_onnx() -> None:
    from . import ort

    ort.prepare()
    weights.tokenizer_file()


def _mojo(device: str) -> Model:
    from . import mojo

    return mojo.load(device)


def _prepare_mojo() -> None:
    from . import mojo

    mojo.build()


BACKENDS = {
    b.name: b
    for b in [
        Backend(
            "pytorch",
            {"gpu": "transformers on PyTorch, fp16, ROCm GPU", "cpu": "transformers on PyTorch, bf16, CPU"},
            _prepare_pytorch,
            lambda device: _pytorch(device, compile=False),
        ),
        Backend(
            "pytorch-compile",
            {
                "gpu": "transformers on PyTorch, fp16, ROCm GPU, encoder through torch.compile",
                "cpu": "transformers on PyTorch, bf16, CPU, encoder through torch.compile",
            },
            _prepare_pytorch,
            lambda device: _pytorch(device, compile=True),
        ),
        Backend(
            "tinygrad",
            {"gpu": "tinygrad, fp16, AMD GPU (no warm-up)", "cpu": "tinygrad, fp16, CPU (no warm-up)"},
            _prepare_tinygrad,
            _tinygrad,
            {"gpu": {"DEV": "AMD"}, "cpu": {"DEV": "CPU"}},
        ),
        Backend(
            "onnx",
            {
                "gpu": "ONNX Runtime, encoder fp16 on WebGPU, decoder and joint on the CPU",
                "cpu": "ONNX Runtime, fp32, CPU",
            },
            _prepare_onnx,
            _onnx,
        ),
        Backend(
            "mojo",
            {
                "gpu": "Mojo, encoder fp16 on the GPU through HIP, decoder and joint on the CPU",
                "cpu": "Mojo, fp32, CPU",
            },
            _prepare_mojo,
            _mojo,
        ),
    ]
}
