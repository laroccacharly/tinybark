"""Dropped attempt: freeze the warm bucketed model to disk.

Pickling the model with its captured TinyJits and restoring it on start would
skip warm-up (about 5.6 s per bucket). Restoring needed workarounds for tinygrad
internals (_buffer and _Tensor below), produced a 1.29 GB pickle on top of the
weights, and wasn't reliable enough to ship: tinybark warms up in memory at start
instead. Kept for reference; not imported by tinybark.

The warm model, frozen to disk: weights, LSTM state and every padded length's
captured JITs, pickled once and restored on start. Restoring skips tinygrad's
Python scheduling, about 5.6 s per padded length."""

import hashlib
import json
import pickle
import weakref
from importlib import metadata
from pathlib import Path

from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.tensor import Tensor, all_tensors

from ... import weights
from .. import model

# Covers dictation up to 30 s.
MAX_SECONDS = 30.0


def tinygrad_commit() -> str:
    info = json.loads(metadata.distribution("tinygrad").read_text("direct_url.json") or "{}")
    return info.get("vcs_info", {}).get("commit_id") or metadata.version("tinygrad")


def key(max_frames: int) -> str:
    """Changes with anything a pickle depends on: the weights, tinygrad, the device,
    tinybark's code and the lengths covered."""
    sources = sorted(Path(__file__).parents[1].glob("*.py"))
    parts = [weights.REVISION, str(weights.FORMAT), tinygrad_commit(), Device.DEFAULT, str(max_frames)]
    digest = hashlib.sha256("\0".join(parts).encode())
    for source in sources:
        digest.update(source.read_bytes())
    return digest.hexdigest()[:16]


def _buffer(device, nbytes, opaque=None, options=None, initial_value=None, *rest) -> Buffer:
    """Buffer's unpickler, waiting for each copy: Buffer() copies its contents from a
    temporary host buffer it frees before the copy is done, which restored wrong
    values into a few small buffers on most loads."""
    buffer = Buffer(device, nbytes, opaque, options, initial_value, *rest)
    if initial_value is not None:
        Device[buffer.device].synchronize()
    return buffer


class _Tensor:
    """Tensor's unpickler, registering each tensor as Tensor() does: tinygrad points
    registered tensors at their realized buffers, while an unregistered one's graph
    grew on every call, each transcription 4 ms slower than the last."""

    def __new__(cls):
        tensor = Tensor.__new__(Tensor)
        all_tensors[weakref.ref(tensor)] = None
        return tensor


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("tinygrad.device", "Buffer"):
            return _buffer
        if (module, name) == ("tinygrad.tensor", "Tensor"):
            return _Tensor
        return super().find_class(module, name)


def load(build, max_seconds: float = MAX_SECONDS):
    """The pickled warm model, or one `build()` returns, warmed up and pickled.
    Pickles of other keys are removed: each holds the 1.25 GB of weights."""
    max_frames = model.bucket(round(max_seconds * 100))
    path = weights.CACHE / f"frozen-{key(max_frames)}.pkl"
    if path.exists():
        with open(path, "rb") as file:
            return _Unpickler(file).load()
    bark = build()
    bark.model.warm_up(max_frames)
    weights.CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as file:
        pickle.dump(bark, file, protocol=5)
    tmp.rename(path)
    for stale in weights.CACHE.glob("frozen-*.pkl"):
        if stale != path:
            stale.unlink()
    return bark
