"""A row-major fp32 matrix and the CPU ops the model needs."""

from std.algorithm import vectorize
from std.math import exp, sqrt, tanh
from std.memory import unsafe_memset_zero
from std.memory.alloc import unsafe_alloc
from std.sys import simd_width_of

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.algorithm import sync_parallelize

comptime W = simd_width_of[DType.float32]()
comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]


struct Tensor(Movable):
    """An owned (rows, cols) fp32 matrix; higher ranks are folded into rows."""

    var ptr: F32Ptr
    var rows: Int
    var cols: Int

    def __init__(out self, rows: Int, cols: Int):
        """Uninitialized storage."""
        self.ptr = unsafe_alloc[Float32](max(rows * cols, 1), alignment=64)
        self.rows = rows
        self.cols = cols

    @staticmethod
    def zeros(rows: Int, cols: Int) -> Tensor:
        var t = Tensor(rows, cols)
        unsafe_memset_zero(t.ptr, rows * cols)
        return t^

    def __deinit__(deinit self):
        self.ptr.unsafe_free()

    def size(self) -> Int:
        return self.rows * self.cols

    def row(self, i: Int) -> F32Ptr:
        return self.ptr.unsafe_offset(i * self.cols)

    def __getitem__(self, i: Int, j: Int) -> Float32:
        return self.ptr[unsafe_offset=i * self.cols + j]

    def __setitem__(mut self, i: Int, j: Int, value: Float32):
        self.ptr[unsafe_offset=i * self.cols + j] = value

    def copy(self) -> Tensor:
        var out = Tensor(self.rows, self.cols)
        for i in range(self.size()):
            out.ptr[unsafe_offset=i] = self.ptr[unsafe_offset=i]
        return out^


struct HalfTensor(Movable):
    """An fp16 copy of a matrix, for matrix-vector products that are bound by
    memory bandwidth: half the bytes, so the joint's token head fits in L3."""

    var ptr: Pointer[Float16, MutUntrackedOrigin]
    var rows: Int
    var cols: Int

    def __init__(out self, t: Tensor):
        self.ptr = unsafe_alloc[Float16](max(t.size(), 1), alignment=64)
        self.rows = t.rows
        self.cols = t.cols
        for i in range(t.size()):
            self.ptr[unsafe_offset=i] = t.ptr[unsafe_offset=i].cast[DType.float16]()

    def __deinit__(deinit self):
        self.ptr.unsafe_free()


def parallel_rows[func: def(Int, Int) -> None](rows: Int, body: func):
    """Calls body(start, end) on contiguous row ranges, one per worker."""
    var workers = min(rows, 12)
    if workers <= 1:
        body(0, rows)
        return
    var chunk = (rows + workers - 1) // workers
    def task(w: Int) {imm}:
        var start = w * chunk
        body(start, min(start + chunk, rows))

    sync_parallelize(task, workers)


def mm(
    c: F32Ptr, a: F32Ptr, b: F32Ptr, m: Int, n: Int, k: Int, transpose_b: Bool
) raises:
    """c (m, n) = a (m, k) @ b, where b is (n, k) if transpose_b else (k, n)."""
    var ct = TileTensor(c.as_unsafe_any_origin(), row_major(m, n))
    var at = TileTensor(a.as_unsafe_any_origin(), row_major(m, k))
    if transpose_b:
        matmul[transpose_b=True](ct, at, TileTensor(b.as_unsafe_any_origin(), row_major(n, k)))
    else:
        matmul(ct, at, TileTensor(b.as_unsafe_any_origin(), row_major(k, n)))


def linear(x: Tensor, w: Tensor) raises -> Tensor:
    """x @ w.T, with w in PyTorch's (out, in) layout."""
    var out = Tensor(x.rows, w.rows)
    mm(out.ptr, x.ptr, w.ptr, x.rows, w.rows, x.cols, True)
    return out^


def linear(x: Tensor, w: Tensor, b: Tensor) raises -> Tensor:
    var out = linear(x, w)
    add_row(out, b)
    return out^


def add_row(mut x: Tensor, b: Tensor):
    """x[i, :] += b for every row."""
    for i in range(x.rows):
        var r = x.row(i)
        def add[w: Int](j: Int) {imm}:
            r.unsafe_store(j, r.unsafe_load[width=w](j) + b.ptr.unsafe_load[width=w](j))

        vectorize[W](x.cols, add)


def add_scaled(mut x: Tensor, y: Tensor, scale: Float32):
    """x += scale * y."""
    def f[w: Int](j: Int) {imm}:
        x.ptr.unsafe_store(j, x.ptr.unsafe_load[width=w](j) + scale * y.ptr.unsafe_load[width=w](j))

    vectorize[W](x.size(), f)


def layernorm(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """Over the last axis, eps 1e-5."""
    var out = Tensor(x.rows, x.cols)
    var n = x.cols
    def rows(start: Int, end: Int) {imm}:
        for i in range(start, end):
            var r = x.row(i)
            var o = out.row(i)
            var total: Float32 = 0
            for j in range(n):
                total += r[unsafe_offset=j]
            var mean = total / Float32(n)
            var var_sum: Float32 = 0
            for j in range(n):
                var d = r[unsafe_offset=j] - mean
                var_sum += d * d
            var inv = 1 / sqrt(var_sum / Float32(n) + 1e-5)
            for j in range(n):
                o[unsafe_offset=j] = (r[unsafe_offset=j] - mean) * inv * w.ptr[unsafe_offset=j] + b.ptr[unsafe_offset=j]

    parallel_rows(x.rows, rows)
    return out^


@inline(.always)
def sigmoid(x: Float32) -> Float32:
    return 1 / (1 + exp(-x))


def silu_(mut x: Tensor):
    def f[w: Int](j: Int) {imm}:
        var v = x.ptr.unsafe_load[width=w](j)
        x.ptr.unsafe_store(j, v / (1 + exp(-v)))

    vectorize[W](x.size(), f)


def relu_(mut x: Tensor):
    def f[w: Int](j: Int) {imm}:
        x.ptr.unsafe_store(j, max(x.ptr.unsafe_load[width=w](j), 0))

    vectorize[W](x.size(), f)


def argmax(x: Tensor, n: Int) -> Int:
    """First index of the maximum of x's first n values, as torch."""
    var p = x.ptr
    var best = 0
    var value = p[unsafe_offset=0]
    for i in range(1, n):
        if p[unsafe_offset=i] > value:
            value = p[unsafe_offset=i]
            best = i
    return best


def gemv(mut y: Tensor, w: Tensor, x: Tensor, xrow: Int, b: Tensor):
    """y[0, :] = w @ x[xrow, :] + b, w (n, k), parallel over output rows."""
    var k = w.cols
    var xp = x.row(xrow)

    def rows(start: Int, end: Int) {imm}:
        for i in range(start, end):
            var r = w.row(i)
            var acc = SIMD[DType.float32, W](0)
            var j = 0
            while j + W <= k:
                acc += r.unsafe_load[width=W](j) * xp.unsafe_load[width=W](j)
                j += W
            var total = acc.reduce_add()
            while j < k:
                total += r[unsafe_offset=j] * xp[unsafe_offset=j]
                j += 1
            y.ptr[unsafe_offset=i] = total + b.ptr[unsafe_offset=i]

    parallel_rows(w.rows, rows)


def gemv(mut y: Tensor, w: HalfTensor, x: Tensor, xrow: Int, b: Tensor):
    """y[0, :] = w @ x[xrow, :] + b with fp16 weights and fp32 math; w.cols is a multiple of W."""
    var k = w.cols
    var xp = x.row(xrow)

    def rows(start: Int, end: Int) {imm}:
        for i in range(start, end):
            var r = w.ptr.unsafe_offset(i * k)
            var acc = SIMD[DType.float32, W](0)
            for j in range(0, k, W):
                acc += r.unsafe_load[width=W](j).cast[DType.float32]() * xp.unsafe_load[width=W](j)
            y.ptr[unsafe_offset=i] = acc.reduce_add() + b.ptr[unsafe_offset=i]

    parallel_rows(w.rows, rows)
