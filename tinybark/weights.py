"""Bark's weights: downloaded from Hugging Face, decoded once with Bark's own
loader, then cached as fp16 safetensors in the layout `model.py` uses."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tinygrad import Tensor, dtypes

# Pinned so a change upstream can't alter what gets downloaded.
REPO = "ConwayResearch/Underdog-Bark-3.0-Clean"
REVISION = "6d34ed8639b2b9a33e764acf06748519bbb950cf"
# Bark is a fine-tune of this model and uses its tokenizer.
TOKENIZER_REPO = "nvidia/parakeet-tdt-0.6b-v3"

CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tinybark"
# Bump when _convert changes, so stale caches aren't loaded.
FORMAT = 2
BN_EPS = 1e-5
# The token head is padded to a multiple of this, for faster matmul and argmax kernels.
VOCAB_ALIGN = 64


def bark_file() -> str:
    return hf_hub_download(REPO, "model.bark", revision=REVISION)


def bark_loader():
    """Bark's loader module, which decodes the entropy-coded weights."""
    path = hf_hub_download(REPO, "bark_loader.py", revision=REVISION)
    spec = importlib.util.spec_from_file_location("bark_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokenizer_file() -> str:
    return hf_hub_download(TOKENIZER_REPO, "tokenizer.json")


def decoded() -> tuple[dict[str, np.ndarray], dict]:
    """Bark's state dict as its loader decodes it, and its config. Decoding the
    entropy-coded weights takes about 15 s, so the result is cached as safetensors."""
    from safetensors.numpy import load_file, save_file

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"bark-{REVISION[:12]}-decoded.safetensors"
    if not path.exists():
        sd, config = bark_loader().load_state_dict(bark_file(), as_numpy=True)
        tmp = path.with_suffix(".tmp")
        save_file({k: np.ascontiguousarray(v) for k, v in sd.items()}, str(tmp), metadata={"config": json.dumps(config)})
        tmp.rename(path)
    with open(path, "rb") as file:
        n = int.from_bytes(file.read(8), "little")
        config = json.loads(json.loads(file.read(n))["__metadata__"]["config"])
    return load_file(str(path)), config


def _convert(sd: dict[str, np.ndarray], config: dict) -> dict[str, np.ndarray]:
    """Fuses q/k/v and each LSTM layer's input and hidden weights, folds each conv
    module's batch norm into its depthwise conv, and splits the joint head into a
    token head (padded with -inf logits) and a duration head."""
    layers, vocab = config["encoder_config"]["num_hidden_layers"], config["vocab_size"]
    out: dict[str, np.ndarray] = {}
    for i in range(layers):
        p = f"encoder.layers.{i}."
        a = p + "self_attn."
        out[a + "qkv"] = np.concatenate([sd.pop(a + f"{n}_proj.weight") for n in "qkv"])
        for n in ("o_proj.weight", "relative_k_proj.weight", "bias_u", "bias_v"):
            out[a + n] = sd.pop(a + n)
        c = p + "conv."
        scale = sd.pop(c + "norm.weight") / np.sqrt(sd.pop(c + "norm.running_var") + BN_EPS)
        out[c + "depthwise.weight"] = sd.pop(c + "depthwise_conv.weight") * scale[:, None, None]
        out[c + "depthwise.bias"] = sd.pop(c + "norm.bias") - sd.pop(c + "norm.running_mean") * scale
        sd.pop(c + "norm.num_batches_tracked", None)
    for l in range(config["num_decoder_layers"]):
        p = "decoder.lstm."
        out[p + f"weight_l{l}"] = np.concatenate([sd.pop(p + f"weight_ih_l{l}"), sd.pop(p + f"weight_hh_l{l}")], 1)
        out[p + f"bias_l{l}"] = sd.pop(p + f"bias_ih_l{l}") + sd.pop(p + f"bias_hh_l{l}")
    weight, bias = sd.pop("joint.head.weight"), sd.pop("joint.head.bias")
    pad = -vocab % VOCAB_ALIGN
    out["joint.token_head.weight"] = np.pad(weight[:vocab], ((0, pad), (0, 0)))
    out["joint.token_head.bias"] = np.pad(bias[:vocab], (0, pad), constant_values=-np.inf)
    out["joint.duration_head.weight"], out["joint.duration_head.bias"] = weight[vocab:], bias[vocab:]
    out.update(sd)
    return {k: np.ascontiguousarray(v) for k, v in out.items()}


def _save(tensors: dict[str, np.ndarray], path: Path) -> None:
    """Writes fp16 safetensors, one tensor at a time to keep memory down."""
    header, offset = {}, 0
    for name, value in tensors.items():
        header[name] = {"dtype": "F16", "shape": list(value.shape), "data_offsets": [offset, offset + value.size * 2]}
        offset += value.size * 2
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as file:
        file.write(len(blob).to_bytes(8, "little") + blob)
        for value in tensors.values():
            file.write(value.astype(np.float16).tobytes())


def load() -> tuple[dict[str, Tensor], dict]:
    """The model's tensors (fp16, on the default device) and its config."""
    CACHE.mkdir(parents=True, exist_ok=True)
    name = f"bark-{REVISION[:12]}-v{FORMAT}"
    path, config_path = CACHE / f"{name}.safetensors", CACHE / f"{name}.json"
    if not path.exists():
        sd, config = decoded()
        sd = {k: v.astype(np.float32) for k, v in sd.items() if v.dtype.kind == "f"}
        sd = _convert(sd, config)
        tmp = path.with_suffix(".tmp")
        _save(sd, tmp)
        del sd
        tmp.rename(path)
        config_path.write_text(json.dumps(config))
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
    return tensors, json.loads(config_path.read_text())
