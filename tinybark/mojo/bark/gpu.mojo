"""The FastConformer encoder layers on the GPU, as tinybark's ONNX Runtime setup
runs them: fp16 matmuls (fp32 accumulation) through MAX's kernels; LayerNorm,
softmax and the residual stream in fp32."""

from std.collections import Dict
from std.math import ceildiv, cos, exp, sin

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives import block

from .model import CHANNELS, HEAD_DIM, HEADS, HIDDEN, KERNEL, LAYERS, N_MELS, subsampled
from .tensor import Tensor
from .weights import Weights

comptime F32 = Pointer[Float32, MutAnyOrigin]
comptime F16 = Pointer[Float16, MutAnyOrigin]
comptime THREADS = 256


# Kernels


def layernorm_kernel[out_type: DType](
    dst: Pointer[Scalar[out_type], MutAnyOrigin], x: F32, w: F32, b: F32, cols_: Int32
):
    """One block per row; may run in place."""
    var cols = Int(cols_)
    var row = block_idx.x
    var src = x.unsafe_offset(row * cols)
    var total: Float32 = 0
    for j in range(thread_idx.x, cols, THREADS):
        total += src[unsafe_offset=j]
    var mean = block.sum[block_size=THREADS](total) / Float32(cols)
    var spread: Float32 = 0
    for j in range(thread_idx.x, cols, THREADS):
        var d = src[unsafe_offset=j] - mean
        spread += d * d
    var inv = 1 / (block.sum[block_size=THREADS](spread) / Float32(cols) + 1e-5) ** 0.5
    for j in range(thread_idx.x, cols, THREADS):
        var v = (src[unsafe_offset=j] - mean) * inv * w[unsafe_offset=j] + b[unsafe_offset=j]
        dst[unsafe_offset=row * cols + j] = v.cast[out_type]()


def silu_half_kernel(dst: F16, x: F32, n_: Int32):
    var n = Int(n_)
    var i = block_idx.x * THREADS + thread_idx.x
    if i < n:
        var v = x[unsafe_offset=i]
        dst[unsafe_offset=i] = (v / (1 + exp(-v))).cast[DType.float16]()


def half_kernel(dst: F16, x: F32, n_: Int32):
    var n = Int(n_)
    var i = block_idx.x * THREADS + thread_idx.x
    if i < n:
        dst[unsafe_offset=i] = x[unsafe_offset=i].cast[DType.float16]()


def residual_kernel(x: F32, y: F32, scale: Float32, n_: Int32):
    var n = Int(n_)
    var i = block_idx.x * THREADS + thread_idx.x
    if i < n:
        x[unsafe_offset=i] += scale * y[unsafe_offset=i]


def bias_kernel(x: F32, b: F32, cols_: Int32, n_: Int32):
    var cols = Int(cols_)
    var n = Int(n_)
    var i = block_idx.x * THREADS + thread_idx.x
    if i < n:
        x[unsafe_offset=i] += b[unsafe_offset=i % cols]


def conv_kernel(dst: F16, pw: F32, w: F32, b: F32, t_: Int32):
    """GLU of pointwise_conv1's output (t, 2 * HIDDEN), depthwise conv along
    time (batch norm folded in), then SiLU, to fp16 (t, HIDDEN)."""
    var t = Int(t_)
    var i = block_idx.x * THREADS + thread_idx.x
    if i >= t * HIDDEN:
        return
    var frame = i // HIDDEN
    var c = i % HIDDEN
    var acc = b[unsafe_offset=c]
    for k in range(KERNEL):
        var s = frame + k - (KERNEL - 1) // 2
        if s >= 0 and s < t:
            var a = pw[unsafe_offset=s * 2 * HIDDEN + c]
            var g = pw[unsafe_offset=s * 2 * HIDDEN + HIDDEN + c]
            acc += w[unsafe_offset=c * KERNEL + k] * a / (1 + exp(-g))
    dst[unsafe_offset=i] = (acc / (1 + exp(-acc))).cast[DType.float16]()


def conv3x3_kernel(dst: F32, x: F32, w: F32, b: F32, channels_: Int32, h_: Int32, w_: Int32, depthwise_: Int32):
    """3x3 convolution, stride 2, padding 1, over channel planes (h, w):
    depthwise, or from one input plane to `channels` (then ReLU)."""
    var channels = Int(channels_)
    var h = Int(h_)
    var wd = Int(w_)
    var depthwise = depthwise_ != 0
    var h2 = (h - 1) // 2 + 1
    var w2 = (wd - 1) // 2 + 1
    var idx = block_idx.x * THREADS + thread_idx.x
    if idx >= channels * h2 * w2:
        return
    var c = idx // (h2 * w2)
    var i = (idx // w2) % h2
    var j = idx % w2
    var src = x.unsafe_offset(c * h * wd if depthwise else 0)
    var acc = b[unsafe_offset=c]
    for di in range(3):
        var y = 2 * i + di - 1
        if y < 0 or y >= h:
            continue
        for dj in range(3):
            var xx = 2 * j + dj - 1
            if xx >= 0 and xx < wd:
                acc += w[unsafe_offset=c * 9 + 3 * di + dj] * src[unsafe_offset=y * wd + xx]
    dst[unsafe_offset=idx] = acc if depthwise else max(acc, 0)


def bias_relu_rows_kernel(x: F32, b: F32, cols_: Int32, n_: Int32):
    """x[r, :] = relu(x[r, :] + b[r])."""
    var cols = Int(cols_)
    var n = Int(n_)
    var i = block_idx.x * THREADS + thread_idx.x
    if i < n:
        x[unsafe_offset=i] = max(x[unsafe_offset=i] + b[unsafe_offset=i // cols], 0)


def flatten_kernel(dst: F16, x: F32, t_: Int32, f_: Int32):
    """(channels, t, f) to (t, channels * f), fp16."""
    var t = Int(t_)
    var f = Int(f_)
    var i = block_idx.x * THREADS + thread_idx.x
    if i >= CHANNELS * t * f:
        return
    var frame = i // (CHANNELS * f)
    var c = (i // f) % CHANNELS
    dst[unsafe_offset=i] = x[unsafe_offset=(c * t + frame) * f + i % f].cast[DType.float16]()


def positions_kernel(dst: F16, t_: Int32):
    """Relative positional encoding for positions t-1 … -(t-1): interleaved sin and cos."""
    var t = Int(t_)
    var i = block_idx.x * THREADS + thread_idx.x
    if i >= (2 * t - 1) * HIDDEN:
        return
    var n = i // HIDDEN
    var c = i % HIDDEN
    var inv_freq = 1 / Float32(10000) ** (Float32(c // 2 * 2) / Float32(HIDDEN))
    var angle = Float32(t - 1 - n) * inv_freq
    dst[unsafe_offset=i] = (sin(angle) if c % 2 == 0 else cos(angle)).cast[DType.float16]()


def heads_kernel(qu: F16, qv: F16, k: F16, v: F16, qkv: F32, bias_u: F32, bias_v: F32, t_: Int32, padded_: Int32):
    """Splits qkv (t, 3 * HIDDEN) into per-head fp16 blocks (HEADS, t, HEAD_DIM):
    the queries plus each positional bias and the keys; the values' blocks have
    `padded` rows."""
    var t = Int(t_)
    var padded = Int(padded_)
    var i = block_idx.x * THREADS + thread_idx.x
    if i >= t * HIDDEN:
        return
    var frame = i // HIDDEN
    var c = i % HIDDEN
    var head = c // HEAD_DIM
    var dst = (head * t + frame) * HEAD_DIM + c % HEAD_DIM
    var src = qkv.unsafe_offset(frame * 3 * HIDDEN + c)
    var q = src[unsafe_offset=0]
    qu[unsafe_offset=dst] = (q + bias_u[unsafe_offset=c]).cast[DType.float16]()
    qv[unsafe_offset=dst] = (q + bias_v[unsafe_offset=c]).cast[DType.float16]()
    k[unsafe_offset=dst] = src[unsafe_offset=HIDDEN].cast[DType.float16]()
    v[unsafe_offset=(head * padded + frame) * HEAD_DIM + c % HEAD_DIM] = src[unsafe_offset=2 * HIDDEN].cast[DType.float16]()


def split_heads_kernel(dst: F16, x: F32, rows_: Int32):
    """(rows, HIDDEN) fp32 to per-head fp16 blocks (HEADS, rows, HEAD_DIM)."""
    var rows = Int(rows_)
    var i = block_idx.x * THREADS + thread_idx.x
    if i >= rows * HIDDEN:
        return
    var row = i // HIDDEN
    var c = i % HIDDEN
    dst[unsafe_offset=((c // HEAD_DIM) * rows + row) * HEAD_DIM + c % HEAD_DIM] = x[unsafe_offset=i].cast[DType.float16]()


def merge_heads_kernel(dst: F16, x: F32, t_: Int32):
    """Per-head blocks (HEADS, t, HEAD_DIM) fp32 to (t, HIDDEN) fp16."""
    var t = Int(t_)
    var i = block_idx.x * THREADS + thread_idx.x
    if i >= t * HIDDEN:
        return
    var frame = i // HIDDEN
    var c = i % HIDDEN
    dst[unsafe_offset=i] = x[unsafe_offset=((c // HEAD_DIM) * t + frame) * HEAD_DIM + c % HEAD_DIM].cast[DType.float16]()


def softmax_kernel(probs: F16, ac: F32, bd: F32, t_: Int32, padded_: Int32):
    """One block per (head, query) row: content scores ac (t) plus position
    scores bd (2t - 1) after the relative shift (Transformer-XL appendix B:
    score (i, j) reads relative position i - j, column t - 1 - i + j), scaled
    and softmaxed to fp16 rows of `padded` columns."""
    var t = Int(t_)
    var padded = Int(padded_)
    var row = block_idx.x
    var i = row % t
    var a = ac.unsafe_offset(row * t)
    var b = bd.unsafe_offset(row * (2 * t - 1) + t - 1 - i)
    var scale = Float32(HEAD_DIM) ** -0.5
    var top = Float32.MIN
    for j in range(thread_idx.x, t, THREADS):
        top = max(top, (a[unsafe_offset=j] + b[unsafe_offset=j]) * scale)
    top = block.max[block_size=THREADS](top)
    var total: Float32 = 0
    for j in range(thread_idx.x, t, THREADS):
        total += exp((a[unsafe_offset=j] + b[unsafe_offset=j]) * scale - top)
    total = block.sum[block_size=THREADS](total)
    for j in range(thread_idx.x, t, THREADS):
        var e = exp((a[unsafe_offset=j] + b[unsafe_offset=j]) * scale - top)
        probs[unsafe_offset=row * padded + j] = (e / total).cast[DType.float16]()


# The encoder


struct GpuEncoder(Movable):
    var ctx: DeviceContext
    # Matrices in fp16, vectors (norms, biases, depthwise kernels) in fp32.
    var mats: Dict[String, DeviceBuffer[DType.float16]]
    var vecs: Dict[String, DeviceBuffer[DType.float32]]
    var out_rows: Int

    def __init__(out self, ctx: DeviceContext, weights: Weights) raises:
        """Uploads the encoder layers' and projector's weights."""
        self.ctx = ctx
        self.mats = {}
        self.vecs = {}
        self.out_rows = 0
        # Vectors are converted to fp32 into one ordinary host array and copied
        # from there: a pinned staging buffer took ~0.4 s to allocate.
        var vec_names = List[String]()
        var vec_total = 0
        for item in weights.entries.items():
            var name = item.key
            if not (name.startswith("encoder.") or name.startswith("encoder_projector.")):
                continue
            var data = weights.fp16(name)
            var n = data[1].rows * data[1].cols
            # Conv kernels (3x3 and 9-tap) and the positional biases are vectors.
            var matrix = data[1].rows > 1 and data[1].cols > 9 and not name.endswith("bias_u") and not name.endswith("bias_v")
            if matrix:
                var buf = ctx.enqueue_create_buffer[DType.float16](n)
                ctx.enqueue_copy(dst_buf=buf, src_ptr=data[0])
                self.mats[name] = buf^
            else:
                vec_names.append(name)
                vec_total += n
            if name == "encoder_projector.weight":
                self.out_rows = data[1].rows
        var host = Tensor(1, vec_total)
        var offset = 0
        for name in vec_names:
            var data = weights.fp16(name)
            var n = data[1].rows * data[1].cols
            for j in range(n):
                host.ptr[unsafe_offset=offset + j] = data[0][unsafe_offset=j].cast[DType.float32]()
            var buf = ctx.enqueue_create_buffer[DType.float32](n)
            ctx.enqueue_copy(dst_buf=buf, src_ptr=host.ptr.unsafe_offset(offset))
            self.vecs[name] = buf^
            offset += n
        ctx.synchronize()
        _ = host^

    def mat(self, name: String) raises -> F16:
        return self.mats[name].unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()

    def vec(self, name: String) raises -> F32:
        return self.vecs[name].unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()

    def linear(self, dst: F32, x: F16, name: String, m: Int, k: Int) raises:
        """dst (m, n) = x (m, k) @ W.T, W the fp16 (n, k) matrix `name`."""
        mm(self.ctx, dst, x, self.mat(name), m, len(self.mats[name]) // k, k, True)

    def layernorm[out_type: DType](
        self, dst: Pointer[Scalar[out_type], MutAnyOrigin], x: F32, p: String, rows: Int
    ) raises:
        self.ctx.enqueue_function[layernorm_kernel[out_type]](
            dst, x, self.vec(p + ".weight"), self.vec(p + ".bias"), Int32(HIDDEN),
            grid_dim=rows, block_dim=THREADS,
        )

    def residual(self, x: F32, y: F32, scale: Float32, n: Int) raises:
        """x += scale * y."""
        self.ctx.enqueue_function[residual_kernel](x, y, scale, Int32(n), grid_dim=ceildiv(n, THREADS), block_dim=THREADS)

    def feed_forward(self, s: Scratch, p: String) raises:
        self.layernorm(s.xn.ptr(), s.x.ptr(), p.replace("feed_forward", "norm_feed_forward"), s.t)
        self.linear(s.hidden.ptr(), s.xn.ptr(), p + ".linear1.weight", s.t, HIDDEN)
        var n = s.t * 4 * HIDDEN
        self.ctx.enqueue_function[silu_half_kernel](s.hidden16.ptr(), s.hidden.ptr(), Int32(n), grid_dim=ceildiv(n, THREADS), block_dim=THREADS)
        self.linear(s.y.ptr(), s.hidden16.ptr(), p + ".linear2.weight", s.t, 4 * HIDDEN)
        self.residual(s.x.ptr(), s.y.ptr(), 0.5, s.t * HIDDEN)

    def attention(self, s: Scratch, p: String) raises:
        var t = s.t
        var n = 2 * t - 1
        var cells = t * HIDDEN
        var blocks = ceildiv(cells, THREADS)
        self.layernorm(s.xn.ptr(), s.x.ptr(), p + "norm_self_att", t)
        self.linear(s.qkv.ptr(), s.xn.ptr(), p + "self_attn.qkv", t, HIDDEN)
        self.linear(s.rel.ptr(), s.pos.ptr(), p + "self_attn.relative_k_proj.weight", n, HIDDEN)
        self.ctx.enqueue_function[heads_kernel](
            s.qu.ptr(), s.qv.ptr(), s.k.ptr(), s.v.ptr(), s.qkv.ptr(),
            self.vec(p + "self_attn.bias_u"), self.vec(p + "self_attn.bias_v"), Int32(t), Int32(s.padded),
            grid_dim=blocks, block_dim=THREADS,
        )
        self.ctx.enqueue_function[split_heads_kernel](
            s.rel16.ptr(), s.rel.ptr(), Int32(n), grid_dim=ceildiv(n * HIDDEN, THREADS), block_dim=THREADS
        )
        for h in range(HEADS):
            var q = h * t * HEAD_DIM
            mm(self.ctx, s.ac.ptr().unsafe_offset(h * t * t), s.qu.ptr().unsafe_offset(q), s.k.ptr().unsafe_offset(q), t, t, HEAD_DIM, True)
            mm(self.ctx, s.bd.ptr().unsafe_offset(h * t * n), s.qv.ptr().unsafe_offset(q), s.rel16.ptr().unsafe_offset(h * n * HEAD_DIM), t, n, HEAD_DIM, True)
        self.ctx.enqueue_function[softmax_kernel](
            s.probs.ptr(), s.ac.ptr(), s.bd.ptr(), Int32(t), Int32(s.padded), grid_dim=HEADS * t, block_dim=THREADS
        )
        for h in range(HEADS):
            mm(
                self.ctx, s.heads.ptr().unsafe_offset(h * t * HEAD_DIM), s.probs.ptr().unsafe_offset(h * t * s.padded),
                s.v.ptr().unsafe_offset(h * s.padded * HEAD_DIM), t, HEAD_DIM, s.padded, False,
            )
        self.ctx.enqueue_function[merge_heads_kernel](s.mixed.ptr(), s.heads.ptr(), Int32(t), grid_dim=blocks, block_dim=THREADS)
        self.linear(s.y.ptr(), s.mixed.ptr(), p + "self_attn.o_proj.weight", t, HIDDEN)
        self.residual(s.x.ptr(), s.y.ptr(), 1, cells)

    def conv(self, s: Scratch, p: String) raises:
        var cells = s.t * HIDDEN
        self.layernorm(s.xn.ptr(), s.x.ptr(), p + "norm_conv", s.t)
        self.linear(s.hidden.ptr(), s.xn.ptr(), p + "conv.pointwise_conv1.weight", s.t, HIDDEN)
        self.ctx.enqueue_function[conv_kernel](
            s.mixed.ptr(), s.hidden.ptr(), self.vec(p + "conv.depthwise.weight"), self.vec(p + "conv.depthwise.bias"), Int32(s.t),
            grid_dim=ceildiv(cells, THREADS), block_dim=THREADS,
        )
        self.linear(s.y.ptr(), s.mixed.ptr(), p + "conv.pointwise_conv2.weight", s.t, HIDDEN)
        self.residual(s.x.ptr(), s.y.ptr(), 1, cells)

    def subsample(self, features: Tensor, s: Scratch) raises:
        """Log-mel features (frames, N_MELS) on the host to s.x (t, HIDDEN)."""
        var ctx = self.ctx
        var p = String("encoder.subsampling.")
        var h = features.rows
        var f = N_MELS
        # `a` holds the features, then each depthwise conv's output; `b` the
        # first conv's output, then each pointwise conv's.
        var cells = CHANNELS * subsampled(h, 1) * subsampled(f, 1)
        var a = Buffer[DType.float32](ctx, max(features.size(), CHANNELS * subsampled(h, 2) * subsampled(f, 2)))
        # Copies as many elements as the destination holds, so give it a view
        # the size of the features.
        ctx.enqueue_copy(dst_buf=a.buf.create_sub_buffer[DType.float32](0, features.size()), src_ptr=features.ptr)
        var b = Buffer[DType.float32](ctx, cells)
        var b16 = Buffer[DType.float16](ctx, cells)
        ctx.enqueue_function[conv3x3_kernel](
            b.ptr(), a.ptr(), self.vec(p + "layers.0.weight"), self.vec(p + "layers.0.bias"), Int32(CHANNELS), Int32(h), Int32(f), Int32(0),
            grid_dim=ceildiv(cells, THREADS), block_dim=THREADS,
        )
        h = subsampled(h, 1)
        f = subsampled(f, 1)
        for i in [2, 5]:
            var dw = p + "layers." + String(i) + "."
            var pw = p + "layers." + String(i + 1) + "."
            var n = CHANNELS * subsampled(h, 1) * subsampled(f, 1)
            ctx.enqueue_function[conv3x3_kernel](
                a.ptr(), b.ptr(), self.vec(dw + "weight"), self.vec(dw + "bias"), Int32(CHANNELS), Int32(h), Int32(f), Int32(1),
                grid_dim=ceildiv(n, THREADS), block_dim=THREADS,
            )
            h = subsampled(h, 1)
            f = subsampled(f, 1)
            ctx.enqueue_function[half_kernel](b16.ptr(), a.ptr(), Int32(n), grid_dim=ceildiv(n, THREADS), block_dim=THREADS)
            # 1x1 conv: weight (out, in) @ planes (in, h * f).
            mm(ctx, b.ptr(), self.mat(pw + "weight"), b16.ptr(), CHANNELS, h * f, CHANNELS, False)
            ctx.enqueue_function[bias_relu_rows_kernel](
                b.ptr(), self.vec(pw + "bias"), Int32(h * f), Int32(n), grid_dim=ceildiv(n, THREADS), block_dim=THREADS
            )
        var n = CHANNELS * h * f
        ctx.enqueue_function[flatten_kernel](b16.ptr(), b.ptr(), Int32(h), Int32(f), grid_dim=ceildiv(n, THREADS), block_dim=THREADS)
        self.linear(s.x.ptr(), b16.ptr(), p + "linear.weight", h, CHANNELS * f)
        ctx.enqueue_function[bias_kernel](
            s.x.ptr(), self.vec(p + "linear.bias"), Int32(HIDDEN), Int32(h * HIDDEN), grid_dim=ceildiv(h * HIDDEN, THREADS), block_dim=THREADS
        )
        ctx.synchronize()
        _ = a^
        _ = b^
        _ = b16^

    def encode(self, features: Tensor) raises -> Tensor:
        """Log-mel features (frames, N_MELS), fp32 on the host, to encoder frames
        projected for the joint (t, decoder hidden), fp32 on the host."""
        var t = subsampled(features.rows, 3)
        var s = Scratch(self.ctx, t, self.out_rows)
        self.subsample(features, s)
        var n = (2 * t - 1) * HIDDEN
        self.ctx.enqueue_function[positions_kernel](s.pos.ptr(), Int32(t), grid_dim=ceildiv(n, THREADS), block_dim=THREADS)
        for i in range(LAYERS):
            var p = "encoder.layers." + String(i) + "."
            self.feed_forward(s, p + "feed_forward1")
            self.attention(s, p)
            self.conv(s, p)
            self.feed_forward(s, p + "feed_forward2")
            self.layernorm(s.x.ptr(), s.x.ptr(), p + "norm_out", t)
        var cells = t * HIDDEN
        self.ctx.enqueue_function[half_kernel](s.xn.ptr(), s.x.ptr(), Int32(cells), grid_dim=ceildiv(cells, THREADS), block_dim=THREADS)
        self.linear(s.encoded.ptr(), s.xn.ptr(), "encoder_projector.weight", t, HIDDEN)
        var m = t * self.out_rows
        self.ctx.enqueue_function[bias_kernel](
            s.encoded.ptr(), self.vec("encoder_projector.bias"), Int32(self.out_rows), Int32(m),
            grid_dim=ceildiv(m, THREADS), block_dim=THREADS,
        )
        var out = Tensor(t, self.out_rows)
        s.encoded.buf.enqueue_copy_to(dst_ptr=out.ptr)
        self.ctx.synchronize()
        _ = s^
        return out^


def mm(ctx: DeviceContext, c: F32, a: F16, b: F16, m: Int, n: Int, k: Int, transpose_b: Bool) raises:
    """c (m, n) = a (m, k) @ b, where b is (n, k) if transpose_b else (k, n)."""
    var ct = TileTensor(c, row_major(m, n))
    var at = TileTensor(a, row_major(m, k))
    if transpose_b:
        matmul[transpose_b=True, target="gpu"](ct, at, TileTensor(b, row_major(n, k)), ctx)
    else:
        matmul[target="gpu"](ct, at, TileTensor(b, row_major(k, n)), ctx)


struct Buffer[dtype: DType](Movable):
    var buf: DeviceBuffer[Self.dtype]

    def __init__(out self, ctx: DeviceContext, n: Int, zero: Bool = False) raises:
        self.buf = ctx.enqueue_create_buffer[Self.dtype](n)
        if zero:
            self.buf.enqueue_fill(0)

    def ptr(self) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
        return self.buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()


struct Scratch(Movable):
    """The encoder's activations for t frames, on the GPU. Per-head blocks are
    (HEADS, rows, HEAD_DIM); probabilities and values are padded to a multiple
    of 16 frames with zeros, as the RDNA matmul kernels need."""

    var t: Int
    var padded: Int
    var x: Buffer[DType.float32]
    var xn: Buffer[DType.float16]
    var y: Buffer[DType.float32]
    var hidden: Buffer[DType.float32]
    var hidden16: Buffer[DType.float16]
    var qkv: Buffer[DType.float32]
    var pos: Buffer[DType.float16]
    var rel: Buffer[DType.float32]
    var rel16: Buffer[DType.float16]
    var qu: Buffer[DType.float16]
    var qv: Buffer[DType.float16]
    var k: Buffer[DType.float16]
    var v: Buffer[DType.float16]
    var ac: Buffer[DType.float32]
    var bd: Buffer[DType.float32]
    var probs: Buffer[DType.float16]
    var heads: Buffer[DType.float32]
    var mixed: Buffer[DType.float16]
    var encoded: Buffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, t: Int, out_cols: Int) raises:
        var n = 2 * t - 1
        self.t = t
        self.padded = (t + 15) // 16 * 16
        self.x = Buffer[DType.float32](ctx, t * HIDDEN)
        self.xn = Buffer[DType.float16](ctx, t * HIDDEN)
        self.y = Buffer[DType.float32](ctx, t * HIDDEN)
        self.hidden = Buffer[DType.float32](ctx, t * 4 * HIDDEN)
        self.hidden16 = Buffer[DType.float16](ctx, t * 4 * HIDDEN)
        self.qkv = Buffer[DType.float32](ctx, t * 3 * HIDDEN)
        self.pos = Buffer[DType.float16](ctx, n * HIDDEN)
        self.rel = Buffer[DType.float32](ctx, n * HIDDEN)
        self.rel16 = Buffer[DType.float16](ctx, n * HIDDEN)
        self.qu = Buffer[DType.float16](ctx, t * HIDDEN)
        self.qv = Buffer[DType.float16](ctx, t * HIDDEN)
        self.k = Buffer[DType.float16](ctx, t * HIDDEN)
        self.v = Buffer[DType.float16](ctx, self.padded * HIDDEN, zero=True)
        self.ac = Buffer[DType.float32](ctx, HEADS * t * t)
        self.bd = Buffer[DType.float32](ctx, HEADS * t * n)
        self.probs = Buffer[DType.float16](ctx, HEADS * t * self.padded, zero=True)
        self.heads = Buffer[DType.float32](ctx, t * HIDDEN)
        self.mixed = Buffer[DType.float16](ctx, t * HIDDEN)
        self.encoded = Buffer[DType.float32](ctx, t * out_cols)
