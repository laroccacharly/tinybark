"""Exports Bark from transformers' ParakeetForTDT to ONNX, in three graphs the
greedy TDT loop in `ort.py` drives: the encoder (fp16 for the GPU, fp32 for the
CPU), one LSTM step and the joint.
Needs torch, transformers and onnxscript (the dev dependencies)."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import ParakeetForTDT

from ..pytorch.model import torch_model


class Encoder(nn.Module):
    """Features (1, frames, mels), unpadded, to encoder frames projected for the joint (frames / 8, hidden).
    Float32 in and out, whatever the model's dtype."""

    def __init__(self, model: ParakeetForTDT):
        super().__init__()
        self.model = model

    def forward(self, features):
        return self.model.get_audio_features(input_features=features.to(self.model.dtype)).pooler_output[0].float()


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
    """The decoder and joint in fp32 for the CPU, the encoder in fp32 for the CPU and in fp16 for the GPU."""
    model = torch_model()
    config = model.config
    layers, hidden = config.num_decoder_layers, config.decoder_hidden_size
    features = ((torch.randn(1, 600, config.encoder_config.num_mel_bins),), {"features": {1: torch.export.Dim("frames", min=9, max=100_000)}})
    # In order: the fp16 encoder last, after model.half().
    graphs = {
        "decoder": (Decoder(model), (torch.zeros(1, 1, dtype=torch.long), torch.zeros(layers, 1, hidden), torch.zeros(layers, 1, hidden)), None),
        "joint": (Joint(model), (torch.randn(hidden), torch.randn(hidden)), None),
        "encoder-fp32": (Encoder(model), *features),
        "encoder": (Encoder(model), *features),
    }
    directory.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for name, (module, args, shapes) in graphs.items():
            if name == "encoder":
                model.half()
            names = list(shapes) if shapes else None
            torch.onnx.export(module.eval(), args, directory / f"{name}.onnx", input_names=names, dynamic_shapes=shapes, dynamo=True, external_data=name.startswith("encoder"))
    keys = ("blank_token_id", "durations", "max_symbols_per_step", "num_decoder_layers", "decoder_hidden_size")
    (directory / "config.json").write_text(json.dumps({k: getattr(config, k) for k in keys}))
