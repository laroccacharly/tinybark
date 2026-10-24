"""Parakeet TDT (FastConformer encoder, LSTM prediction network, TDT joint) on the
CPU in fp32, a port of tinybark's model.py matching transformers' ParakeetForTDT
with greedy generate().

The encoder runs on the exact input length, so tinybark's padding masks are all
ones and dropped. The joint runs only for the frame the greedy loop reads."""

from std.math import cos, exp, sin, sqrt, tanh

from .tensor import (
    HalfTensor,
    Tensor,
    W,
    add_row,
    add_scaled,
    argmax,
    gemv,
    layernorm,
    linear,
    mm,
    parallel_rows,
    relu_,
    sigmoid,
    silu_,
)
from .weights import Weights

# From the model's config.json.
comptime LAYERS = 24
comptime HIDDEN = 1024
comptime HEADS = 8
comptime HEAD_DIM = HIDDEN // HEADS
comptime KERNEL = 9
comptime CHANNELS = 256
comptime N_MELS = 128
comptime VOCAB = 8193
comptime BLANK = 8192
comptime DECODER_LAYERS = 2
comptime DECODER_HIDDEN = 640
comptime MAX_SYMBOLS = 10
comptime DURATIONS = 5
# Token <|endoftext|>, generation_config.json's eos_token_id: generate() stops on it.
comptime EOS = 3


def subsampled(frames: Int, convs: Int) -> Int:
    """Length after `convs` stride-2 convolutions with kernel 3 and padding 1."""
    var n = frames
    for _ in range(convs):
        n = (n - 1) // 2 + 1
    return n


def positions(length: Int) -> Tensor:
    """Relative positional encoding for positions length-1 … -(length-1): interleaved sin and cos."""
    var out = Tensor(2 * length - 1, HIDDEN)
    for i in range(HIDDEN // 2):
        var inv_freq = 1.0 / (10000.0 ** (Float64(2 * i) / Float64(HIDDEN)))
        for n in range(2 * length - 1):
            var angle = Float64(length - 1 - n) * inv_freq
            out[n, 2 * i] = Float32(sin(angle))
            out[n, 2 * i + 1] = Float32(cos(angle))
    return out^


def conv3x3_s2(
    x: Tensor, h: Int, w: Int, weight: Tensor, bias: Tensor, depthwise: Bool
) -> Tensor:
    """3x3 convolution, stride 2, padding 1, over channel rows of (h, w) planes.
    Depthwise, or (if not) from a single input plane to weight.rows channels."""
    var h2 = (h - 1) // 2 + 1
    var w2 = (w - 1) // 2 + 1
    var out = Tensor(weight.rows, h2 * w2)

    def channels(start: Int, end: Int) {imm}:
        for c in range(start, end):
            var src = x.row(c if depthwise else 0)
            var dst = out.row(c)
            var k = weight.row(c)
            for i in range(h2):
                for j in range(w2):
                    var acc = bias.ptr[unsafe_offset=c]
                    for di in range(3):
                        var y = 2 * i + di - 1
                        if y < 0 or y >= h:
                            continue
                        for dj in range(3):
                            var xx = 2 * j + dj - 1
                            if xx >= 0 and xx < w:
                                acc += k[unsafe_offset=3 * di + dj] * src[unsafe_offset=y * w + xx]
                    dst[unsafe_offset=i * w2 + j] = acc

    parallel_rows(weight.rows, channels)
    return out^


def pointwise(x: Tensor, weight: Tensor, bias: Tensor) raises -> Tensor:
    """1x1 convolution over channel rows: weight (out, in) @ x (in, pixels) + bias per channel, then ReLU."""
    var out = Tensor(weight.rows, x.cols)
    mm(out.ptr, weight.ptr, x.ptr, weight.rows, x.cols, weight.cols, False)
    for c in range(out.rows):
        var r = out.row(c)
        var b = bias.ptr[unsafe_offset=c]
        for j in range(out.cols):
            r[unsafe_offset=j] = max(r[unsafe_offset=j] + b, 0)
    return out^


struct Parakeet(Movable):
    var w: Weights
    # fp16 copies of the decoder's and joint's large matrices: their
    # matrix-vector products are bound by memory bandwidth.
    var token_head: HalfTensor
    var lstm: List[HalfTensor]
    var projector: HalfTensor

    def __init__(out self, var weights: Weights) raises:
        self.token_head = HalfTensor(weights["joint.token_head.weight"])
        self.lstm = []
        for l in range(DECODER_LAYERS):
            self.lstm.append(HalfTensor(weights["decoder.lstm.weight_l" + String(l)]))
        self.projector = HalfTensor(weights["decoder.decoder_projector.weight"])
        self.w = weights^

    # Encoder

    def subsample(self, features: Tensor) raises -> Tensor:
        var p = String("encoder.subsampling.")
        var t = features.rows
        var x = conv3x3_s2(features, t, N_MELS, self.w[p + "layers.0.weight"], self.w[p + "layers.0.bias"], False)
        relu_(x)
        var h = subsampled(t, 1)
        var f = N_MELS // 2
        for i in [2, 5]:
            var dw = p + "layers." + String(i) + "."
            var pw = p + "layers." + String(i + 1) + "."
            var d = conv3x3_s2(x, h, f, self.w[dw + "weight"], self.w[dw + "bias"], True)
            h = (h - 1) // 2 + 1
            f = (f - 1) // 2 + 1
            x = pointwise(d, self.w[pw + "weight"], self.w[pw + "bias"])
        # (channels, time, freq) to (time, channels * freq).
        var flat = Tensor(h, CHANNELS * f)
        for c in range(CHANNELS):
            for i in range(h):
                for j in range(f):
                    flat[i, c * f + j] = x[c, i * f + j]
        return linear(flat, self.w[p + "linear.weight"], self.w[p + "linear.bias"])

    def attention(self, x: Tensor, p: String, pos: Tensor) raises -> Tensor:
        var t = x.rows
        var qkv = linear(x, self.w[p + "qkv"])
        var rel = linear(pos, self.w[p + "relative_k_proj.weight"])
        ref bias_u = self.w[p + "bias_u"]
        ref bias_v = self.w[p + "bias_v"]
        var out = Tensor(t, HIDDEN)
        var scale = Float32(HEAD_DIM) ** -0.5
        comptime V = SIMD[DType.float32, W]

        # One task per (head, query frame): scores, softmax, weighted sum of values.
        def rows(start: Int, end: Int) {imm}:
            var qu = Tensor(1, HEAD_DIM)
            var qv = Tensor(1, HEAD_DIM)
            var scores = Tensor(1, t)
            for item in range(start, end):
                var hd = item // t
                var i = item % t
                var o = hd * HEAD_DIM
                var q = qkv.row(i).unsafe_offset(o)
                for d in range(0, HEAD_DIM, W):
                    var qd = q.unsafe_load[width=W](d)
                    qu.ptr.unsafe_store(d, qd + bias_u.row(hd).unsafe_load[width=W](d))
                    qv.ptr.unsafe_store(d, qd + bias_v.row(hd).unsafe_load[width=W](d))
                var top = Float32.MIN
                for j in range(t):
                    var k = qkv.row(j).unsafe_offset(HIDDEN + o)
                    # Relative shift (Transformer-XL appendix B): score (i, j) reads
                    # relative position i - j, row t - 1 - i + j of `rel`.
                    var r = rel.row(t - 1 - i + j).unsafe_offset(o)
                    var acc = V(0)
                    for d in range(0, HEAD_DIM, W):
                        acc += qu.ptr.unsafe_load[width=W](d) * k.unsafe_load[width=W](d)
                        acc += qv.ptr.unsafe_load[width=W](d) * r.unsafe_load[width=W](d)
                    var s = acc.reduce_add() * scale
                    scores.ptr[unsafe_offset=j] = s
                    top = max(top, s)
                var total: Float32 = 0
                for j in range(t):
                    var e = exp(scores.ptr[unsafe_offset=j] - top)
                    scores.ptr[unsafe_offset=j] = e
                    total += e
                var dst = out.row(i).unsafe_offset(o)
                for d in range(0, HEAD_DIM, W):
                    var acc = V(0)
                    for j in range(t):
                        acc += scores.ptr[unsafe_offset=j] * qkv.row(j).unsafe_offset(2 * HIDDEN + o).unsafe_load[width=W](d)
                    dst.unsafe_store(d, acc / total)

        parallel_rows(HEADS * t, rows)
        return linear(out, self.w[p + "o_proj.weight"])

    def conv(self, x: Tensor, p: String) raises -> Tensor:
        var t = x.rows
        var pw = linear(x, self.w[p + "pointwise_conv1.weight"])
        var glu = Tensor(t, HIDDEN)
        for i in range(t):
            for c in range(HIDDEN):
                glu[i, c] = pw[i, c] * sigmoid(pw[i, HIDDEN + c])
        ref weight = self.w[p + "depthwise.weight"]
        ref bias = self.w[p + "depthwise.bias"]
        var dw = Tensor(t, HIDDEN)
        var half = (KERNEL - 1) // 2

        def frames(start: Int, end: Int) {imm}:
            for i in range(start, end):
                for c in range(HIDDEN):
                    var acc = bias.ptr[unsafe_offset=c]
                    for j in range(KERNEL):
                        var s = i + j - half
                        if s >= 0 and s < t:
                            acc += weight[c, j] * glu[s, c]
                    dw.ptr[unsafe_offset=i * HIDDEN + c] = acc

        parallel_rows(t, frames)
        silu_(dw)
        return linear(dw, self.w[p + "pointwise_conv2.weight"])

    def feed_forward(self, x: Tensor, p: String) raises -> Tensor:
        var hidden = linear(x, self.w[p + "linear1.weight"])
        silu_(hidden)
        return linear(hidden, self.w[p + "linear2.weight"])

    def norm(self, x: Tensor, p: String) raises -> Tensor:
        return layernorm(x, self.w[p + ".weight"], self.w[p + ".bias"])

    def encode(self, features: Tensor) raises -> Tensor:
        """Encoder frames projected for the joint, shape (frames, decoder hidden)."""
        var x = self.subsample(features)
        var pos = positions(x.rows)
        for i in range(LAYERS):
            var p = "encoder.layers." + String(i) + "."
            add_scaled(x, self.feed_forward(self.norm(x, p + "norm_feed_forward1"), p + "feed_forward1."), 0.5)
            add_scaled(x, self.attention(self.norm(x, p + "norm_self_att"), p + "self_attn.", pos), 1.0)
            add_scaled(x, self.conv(self.norm(x, p + "norm_conv"), p + "conv."), 1.0)
            add_scaled(x, self.feed_forward(self.norm(x, p + "norm_feed_forward2"), p + "feed_forward2."), 0.5)
            x = self.norm(x, p + "norm_out")
        return linear(x, self.w["encoder_projector.weight"], self.w["encoder_projector.bias"])

    # Prediction network and joint

    def predict(self, token: Int, mut h: Tensor, mut c: Tensor) raises -> Tensor:
        """One LSTM step on `token`, updating the LSTM state `h` and `c` (layers,
        hidden), then the decoder projection, shape (1, decoder hidden)."""
        var x = Tensor(1, 2 * DECODER_HIDDEN)
        ref embedding = self.w["decoder.embedding.weight"]
        for j in range(DECODER_HIDDEN):
            x[0, j] = embedding[token, j]
        var gates = Tensor(1, 4 * DECODER_HIDDEN)
        var out = Tensor(1, DECODER_HIDDEN)
        for l in range(DECODER_LAYERS):
            for j in range(DECODER_HIDDEN):
                x[0, DECODER_HIDDEN + j] = h[l, j]
            gemv(gates, self.lstm[l], x, 0, self.w["decoder.lstm.bias_l" + String(l)])
            for j in range(DECODER_HIDDEN):
                var i = sigmoid(gates[0, j])
                var f = sigmoid(gates[0, DECODER_HIDDEN + j])
                var g = tanh(gates[0, 2 * DECODER_HIDDEN + j])
                var o = sigmoid(gates[0, 3 * DECODER_HIDDEN + j])
                var cell = f * c[l, j] + i * g
                c[l, j] = cell
                h[l, j] = o * tanh(cell)
                x[0, j] = h[l, j]
        gemv(out, self.projector, x, 0, self.w["decoder.decoder_projector.bias"])
        return out^

    def joint(self, encoded: Tensor, frame: Int, decoded: Tensor) raises -> Tuple[Int, Int]:
        """The best token and the best duration index at `frame`."""
        var x = Tensor(1, DECODER_HIDDEN)
        for j in range(DECODER_HIDDEN):
            x[0, j] = max(encoded[frame, j] + decoded[0, j], 0)
        var logits = Tensor(1, self.token_head.rows)
        gemv(logits, self.token_head, x, 0, self.w["joint.token_head.bias"])
        var durations = Tensor(1, DURATIONS)
        gemv(durations, self.w["joint.duration_head.weight"], x, 0, self.w["joint.duration_head.bias"])
        return (argmax(logits, self.token_head.rows), argmax(durations, DURATIONS))

    # Transcription

    def generate(self, features: Tensor) raises -> List[Int]:
        """Greedy TDT decoding on the CPU, as ParakeetForTDT.generate()."""
        if features.rows < 2:
            return []
        return self.decode(self.encode(features), features.rows)

    def decode(self, encoded: Tensor, frames: Int) raises -> List[Int]:
        """The greedy TDT loop over encoder output, for `frames` feature frames."""
        var out = List[Int]()
        var valid = encoded.rows
        var h = Tensor.zeros(DECODER_LAYERS, DECODER_HIDDEN)
        var c = Tensor.zeros(DECODER_LAYERS, DECODER_HIDDEN)
        # generate() starts from the blank token, reuses the decoder's output while
        # it predicts blanks, and sizes its output by the padded feature length.
        var decoded = self.predict(BLANK, h, c)
        var frame = 0
        for _ in range(MAX_SYMBOLS * subsampled(frames + 1, 3) - 1):
            if frame >= valid:
                break
            var best = self.joint(encoded, frame, decoded)
            var token = best[0]
            var duration = best[1]
            if token == BLANK:
                frame += max(duration, 1)
                continue
            out.append(token)
            if token == EOS:
                break
            decoded = self.predict(token, h, c)
            frame += duration
        return out^
