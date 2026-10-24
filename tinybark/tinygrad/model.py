"""Parakeet TDT (FastConformer encoder, LSTM prediction network, TDT joint) in
tinygrad, matching transformers' ParakeetForTDT with greedy generate().

Audio is padded to a few fixed lengths ("buckets") and masked, so each bucket
compiles once and then replays through TinyJit."""

import math

import numpy as np
from tinygrad import Tensor, TinyJit, Variable, dtypes
from tinygrad.uop.ops import UOp

from .. import audio

# Audio is padded to a multiple of this many feature frames (10 ms each), 2.56 s.
BUCKET = 256


def bucket(frames: int) -> int:
    return math.ceil(frames / BUCKET) * BUCKET


def subsampled(frames: int, convs: int) -> int:
    """Length after `convs` stride-2 convolutions with kernel 3 and padding 1."""
    for _ in range(convs):
        frames = (frames - 1) // 2 + 1
    return frames


def positions(length: int, hidden: int) -> np.ndarray:
    """Relative positional encoding for positions length-1 … -(length-1): interleaved sin and cos."""
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, hidden, 2, dtype=np.float32) / hidden))
    freqs = np.arange(length - 1, -length, -1, dtype=np.float32)[:, None] * inv_freq
    return np.stack([np.sin(freqs), np.cos(freqs)], -1).reshape(2 * length - 1, hidden)


def linear(x: Tensor, w: Tensor, b: Tensor | None = None) -> Tensor:
    """fp16 matmul, result in fp32."""
    y = x.half() @ w.T
    return (y if b is None else y + b).float()


def argmax(x: Tensor, group: int = 64) -> Tensor:
    """argmax over the last axis (a multiple of `group`) in two stages: a single
    reduction over thousands of logits runs nearly serially. Ties go to the first
    index, as in torch."""
    rows, n = x.shape
    x = x.reshape(rows, n // group, group)
    # contiguous(): fused into the final sum, each row would rescan all its logits serially.
    inner = x.argmax(-1).contiguous()
    best = x.max(-1).argmax(-1)
    picked = (Tensor.arange(n // group).reshape(1, -1) == best.unsqueeze(1)).where(inner, 0).sum(-1)
    return (best * group + picked).cast(dtypes.int32)


def layernorm(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return x.float().layernorm(eps=1e-5) * w.float() + b.float()


class Parakeet:
    def __init__(self, weights: dict[str, Tensor], config: dict):
        self.w = weights
        enc = config["encoder_config"]
        self.layers = enc["num_hidden_layers"]
        self.hidden = enc["hidden_size"]
        self.heads = enc["num_attention_heads"]
        self.kernel = enc["conv_kernel_size"]
        self.vocab = config["vocab_size"]
        self.blank = config["blank_token_id"]
        self.durations = config["durations"]
        self.decoder_layers = config["num_decoder_layers"]
        self.decoder_hidden = config["decoder_hidden_size"]
        self.max_symbols = config["max_symbols_per_step"]
        # Token <|endoftext|>, generation_config.json's eos_token_id: generate() stops on it.
        self.eos = 3
        self._encoders: dict[int, TinyJit] = {}
        self._steps: dict[int, TinyJit] = {}
        self._positions: dict[int, list[Tensor]] = {}
        # The LSTM state stays on the GPU: building inputs from NumPy costs more than a step.
        self._h = Tensor.zeros(self.decoder_layers, self.decoder_hidden).contiguous().realize()
        self._c = Tensor.zeros(self.decoder_layers, self.decoder_hidden).contiguous().realize()
        self._token = Variable("token", 0, self.vocab - 1)
        self._lengths = [Variable(f"length{i}", 1, 1 << 20) for i in (1, 2, 3)]

    # Encoder

    def _subsample(self, x: Tensor, masks: tuple[Tensor, ...]) -> Tensor:
        w, p = self.w, "encoder.subsampling."
        x = x.unsqueeze(1).half()
        x = (x.conv2d(w[p + "layers.0.weight"], w[p + "layers.0.bias"], stride=2, padding=1) * masks[0].half()).relu()
        for i, mask in zip((2, 5), masks[1:]):
            dw, pw = f"{p}layers.{i}.", f"{p}layers.{i + 1}."
            x = x.conv2d(w[dw + "weight"], w[dw + "bias"], groups=x.shape[1], stride=2, padding=1) * mask.half()
            x = (x.conv2d(w[pw + "weight"], w[pw + "bias"]) * mask.half()).relu()
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        return linear(x, w[p + "linear.weight"], w[p + "linear.bias"])

    def _attention(self, x: Tensor, p: str, rel: Tensor, key_mask: Tensor) -> Tensor:
        w = self.w
        b, t, _ = x.shape
        h = self.heads
        d = self.hidden // h
        qkv = linear(x, w[p + "qkv"]).reshape(b, t, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        bias_u = w[p + "bias_u"].float().reshape(1, h, 1, d)
        bias_v = w[p + "bias_v"].float().reshape(1, h, 1, d)
        ac = (q + bias_u).half() @ k.half().transpose(-1, -2)
        bd = (q + bias_v).half() @ rel
        # Relative shift (Transformer-XL appendix B), as ParakeetEncoderAttention._rel_shift.
        n = bd.shape[-1]
        bd = bd.pad((None, None, None, (1, 0))).reshape(b, h, n + 1, t)[:, :, 1:].reshape(b, h, t, n)[..., :t]
        scores = (ac.float() + bd.float()) * d**-0.5 + key_mask
        out = scores.softmax(-1).half() @ v.half()
        out = out.permute(0, 2, 1, 3).reshape(b, t, self.hidden)
        return linear(out, w[p + "o_proj.weight"])

    def _conv(self, x: Tensor, p: str, time_mask: Tensor) -> Tensor:
        w = self.w
        x = linear(x, w[p + "pointwise_conv1.weight"].reshape(2 * self.hidden, self.hidden))
        x = x[..., : self.hidden] * x[..., self.hidden :].sigmoid()
        x = (x * time_mask).permute(0, 2, 1)
        x = x.conv2d(w[p + "depthwise.weight"].float(), w[p + "depthwise.bias"].float(), groups=self.hidden, padding=(self.kernel - 1) // 2)
        x = x.permute(0, 2, 1).silu()
        return linear(x, w[p + "pointwise_conv2.weight"].reshape(self.hidden, self.hidden))

    def _feed_forward(self, x: Tensor, p: str) -> Tensor:
        return linear(linear(x, self.w[p + "linear1.weight"]).silu(), self.w[p + "linear2.weight"])

    def _relative_keys(self, t: int) -> list[Tensor]:
        """Each layer's projected positional encodings, shape (1, heads, head_dim, 2t - 1).
        They depend only on the length, so they're computed once per bucket."""
        if t not in self._positions:
            pos = Tensor(positions(t, self.hidden))
            h, d = self.heads, self.hidden // self.heads
            rel = [linear(pos, self.w[f"encoder.layers.{i}.self_attn.relative_k_proj.weight"]).half() for i in range(self.layers)]
            self._positions[t] = [r.reshape(1, -1, h, d).permute(0, 2, 3, 1).contiguous().realize() for r in rel]
        return self._positions[t]

    def _encode(self, features: Tensor, *lengths: UOp) -> Tensor:
        """Encoder frames projected for the joint, shape (frames, decoder hidden). `lengths`
        are bound Variables: the valid length after each of the three subsampling convs."""
        w = self.w
        sizes = [subsampled(features.shape[1], i) for i in (1, 2, 3)]
        masks = [(Tensor.arange(s) < n).float().reshape(1, 1, s, 1) for n, s in zip(lengths, sizes)]
        t = sizes[-1]
        key_mask = (Tensor.arange(t) < lengths[-1]).where(0.0, -math.inf).reshape(1, 1, 1, t)
        rel = self._relative_keys(t)
        x = self._subsample(features, masks)
        time_mask = masks[-1].reshape(1, -1, 1)
        for i in range(self.layers):
            p = f"encoder.layers.{i}."
            x = x + 0.5 * self._feed_forward(layernorm(x, w[p + "norm_feed_forward1.weight"], w[p + "norm_feed_forward1.bias"]), p + "feed_forward1.")
            x = x + self._attention(layernorm(x, w[p + "norm_self_att.weight"], w[p + "norm_self_att.bias"]), p + "self_attn.", rel[i], key_mask)
            x = x + self._conv(layernorm(x, w[p + "norm_conv.weight"], w[p + "norm_conv.bias"]), p + "conv.", time_mask)
            x = x + 0.5 * self._feed_forward(layernorm(x, w[p + "norm_feed_forward2.weight"], w[p + "norm_feed_forward2.bias"]), p + "feed_forward2.")
            x = layernorm(x, w[p + "norm_out.weight"], w[p + "norm_out.bias"])
        return linear(x[0], w["encoder_projector.weight"], w["encoder_projector.bias"]).half().realize()

    # Prediction network and joint

    def _step(self, token: UOp, encoded: Tensor) -> Tensor:
        """One LSTM step on `token` (a bound Variable), updating the LSTM state in
        place, then the joint for every encoder frame: shape (2, frames), the best
        token and the best duration index per frame."""
        w, p = self.w, "decoder."
        x = w[p + "embedding.weight"][token : token + 1].float()
        hs, cs = [], []
        for l in range(self.decoder_layers):
            gates = linear(x.cat(self._h[l : l + 1], dim=1), w[p + f"lstm.weight_l{l}"], w[p + f"lstm.bias_l{l}"])
            i, f, g, o = gates.chunk(4, dim=-1)
            cl = f.sigmoid() * self._c[l : l + 1] + i.sigmoid() * g.tanh()
            x = o.sigmoid() * cl.tanh()
            hs.append(x)
            cs.append(cl)
        decoded = linear(x, w[p + "decoder_projector.weight"], w[p + "decoder_projector.bias"])
        joint = (encoded.float() + decoded).relu()
        tokens = argmax(linear(joint, w["joint.token_head.weight"], w["joint.token_head.bias"]))
        durations = linear(joint, w["joint.duration_head.weight"], w["joint.duration_head.bias"]).argmax(-1).cast(dtypes.int32)
        out = tokens.stack(durations)
        # One realize, so the LSTM isn't recomputed for each output.
        Tensor.realize(self._h.assign(Tensor.cat(*hs)), self._c.assign(Tensor.cat(*cs)), out)
        return out

    # Transcription

    def warm_up(self, max_frames: int) -> None:
        """Captures the encoder and decoder-step JITs of every padded length up to
        `max_frames`: TinyJit runs a function on its first call and captures it on
        the second."""
        for size in range(BUCKET, bucket(max_frames) + 1, BUCKET):
            silence = np.zeros(size * audio.HOP, np.float32)
            for _ in range(2):
                self.generate(silence)

    def generate(self, samples: np.ndarray) -> list[int]:
        """Greedy TDT decoding of 16 kHz mono float32 samples, as ParakeetForTDT.generate()."""
        feats = audio.features(samples)
        frames = len(feats)
        if frames < 2:
            return []
        size = bucket(frames)
        padded = np.zeros((1, size, feats.shape[1]), np.float32)
        padded[0, :frames] = feats

        lengths = [subsampled(frames, i) for i in (1, 2, 3)]
        valid = lengths[-1]
        self._relative_keys(subsampled(size, 3))
        encoder = self._encoders.setdefault(size, TinyJit(self._encode))
        encoded = encoder(Tensor(padded), *(v.bind(n) for v, n in zip(self._lengths, lengths)))

        step = self._steps.setdefault(size, TinyJit(self._step))
        Tensor.realize(self._h.assign(Tensor.zeros_like(self._h)), self._c.assign(Tensor.zeros_like(self._c)))

        def predict(token: int) -> np.ndarray:
            return step(self._token.bind(token), encoded).numpy()

        # generate() starts from the blank token, reuses the decoder's output while
        # it predicts blanks, and sizes its output by the padded feature length.
        tokens, durations = predict(self.blank)
        out, frame = [], 0
        for _ in range(self.max_symbols * subsampled(frames + 1, 3) - 1):
            if frame >= valid:
                break
            token, duration = int(tokens[frame]), self.durations[durations[frame]]
            if token == self.blank:
                frame += max(duration, 1)
                continue
            out.append(token)
            if token == self.eos:
                break
            tokens, durations = predict(token)
            frame += duration
        return out
