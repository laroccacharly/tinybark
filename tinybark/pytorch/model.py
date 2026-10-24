"""Bark in transformers' ParakeetForTDT, as stt's bark_loader.load_model builds it."""

import torch
from transformers import GenerationConfig, ParakeetForTDT, ParakeetTDTConfig

from .. import weights


def torch_model(dtype: torch.dtype = torch.float32) -> ParakeetForTDT:
    """Bark in transformers, as bark_loader.load_model builds it, from the decoded-weights cache."""
    sd, config = weights.decoded()
    model = ParakeetForTDT(ParakeetTDTConfig.from_dict(config))
    missing, unexpected = model.load_state_dict({k: torch.from_numpy(v).float() for k, v in sd.items()}, strict=False)
    assert not unexpected and all("num_batches_tracked" in m for m in missing), (missing[:5], unexpected[:5])
    model.generation_config = GenerationConfig.from_pretrained(weights.TOKENIZER_REPO)
    return model.eval().to(dtype=dtype)
