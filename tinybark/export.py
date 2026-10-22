"""Exports Bark from transformers' ParakeetForTDT to ONNX, in three graphs the
greedy TDT loop in `ort.py` drives: the encoder, one LSTM step and the joint.
Needs torch, transformers and onnxscript (the dev dependencies)."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import GenerationConfig, ParakeetForTDT, ParakeetTDTConfig

from . import weights


def torch_model(dtype: torch.dtype = torch.float32) -> ParakeetForTDT:
    """Bark in transformers, as bark_loader.load_model builds it, from the decoded-weights cache."""
    sd, config = weights.decoded()
    model = ParakeetForTDT(ParakeetTDTConfig.from_dict(config))
    missing, unexpected = model.load_state_dict({k: torch.from_numpy(v).float() for k, v in sd.items()}, strict=False)
    assert not unexpected and all("num_batches_tracked" in m for m in missing), (missing[:5], unexpected[:5])
    model.generation_config = GenerationConfig.from_pretrained(weights.TOKENIZER_REPO)
    return model.eval().to(dtype=dtype)


class Encoder(nn.Module):
    """Features (1, frames, mels), unpadded, to encoder frames projected for the joint (frames / 8, hidden)."""

    def __init__(self, model: ParakeetForTDT):
        super().__init__()
        self.model = model

    def forward(self, features):
        return self.model.get_audio_features(input_features=features).pooler_output[0]


class Decoder(nn.Module):
    """One LSTM step: token (1, 1) and state (layers, 1, hidden) to the decoder output (hidden,) and new state."""

    def __init__(self, model: ParakeetForTDT):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, token, h, c):
        out, (h, c) = self.decoder.lstm(self.decoder.embedding(token), (h, c))
        return self.decoder.decoder_projector(out)[0, 0], h, c


class Joint(nn.Module):
    """One encoder frame and the decoder output to the best token and the best duration index."""

    def __init__(self, model: ParakeetForTDT):
        super().__init__()
        self.joint = model.joint
        self.vocab = model.config.vocab_size

    def forward(self, encoded, decoded):
        logits = self.joint(decoder_hidden_states=decoded, encoder_hidden_states=encoded)
        return logits[: self.vocab].argmax(), logits[self.vocab :].argmax()


def export(directory: Path) -> None:
    model = torch_model()
    config = model.config
    layers, hidden = config.num_decoder_layers, config.decoder_hidden_size
    graphs = {
        "encoder": (Encoder(model), (torch.randn(1, 600, config.encoder_config.num_mel_bins),), {"features": {1: torch.export.Dim("frames", min=9, max=100_000)}}),
        "decoder": (Decoder(model), (torch.zeros(1, 1, dtype=torch.long), torch.zeros(layers, 1, hidden), torch.zeros(layers, 1, hidden)), None),
        "joint": (Joint(model), (torch.randn(hidden), torch.randn(hidden)), None),
    }
    directory.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for name, (module, args, shapes) in graphs.items():
            names = list(shapes) if shapes else None
            torch.onnx.export(module.eval(), args, directory / f"{name}.onnx", input_names=names, dynamic_shapes=shapes, dynamo=True, external_data=name == "encoder")
    keys = ("blank_token_id", "durations", "max_symbols_per_step", "num_decoder_layers", "decoder_hidden_size")
    (directory / "config.json").write_text(json.dumps({k: getattr(config, k) for k in keys}))
