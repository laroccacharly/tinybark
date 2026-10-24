"""Bark's converted weights, uploaded to tinygrad's default device."""

import json

from tinygrad import Tensor, dtypes

from .. import weights


def load() -> tuple[dict[str, Tensor], dict]:
    """The model's tensors (fp16, on the default device) and its config."""
    path, config = weights.converted()
    # One read of the whole data block, then one realize that assigns each tensor
    # its own buffer: tinygrad's per-tensor disk reads take ~10 s, and kernels
    # reading views into one big buffer are slower.
    disk = Tensor.empty(path.stat().st_size, dtype=dtypes.uint8, device=f"disk:{path}")
    n = int.from_bytes(disk[:8].data(), "little")
    header = json.loads(bytes(disk[8 : 8 + n].data()))
    data = disk[8 + n :].bitcast(dtypes.half).to(None).realize()
    tensors = {}
    for name, info in header.items():
        start, end = (o // 2 for o in info["data_offsets"])
        tensors[name] = Tensor.empty(*info["shape"], dtype=dtypes.half).assign(data[start:end].reshape(info["shape"]))
    Tensor.realize(*tensors.values())
    del data
    return tensors, config
