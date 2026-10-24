"""tinybark's cached fp16 safetensors (~/.cache/tinybark): fp32 tensors for the
CPU, and the raw fp16 data for uploading to the GPU."""

from std.collections import Dict
from std.memory.alloc import unsafe_alloc
from std.os import getenv
from std.python import Python

from .tensor import Tensor, W

comptime NAME = "bark-6d34ed8639b2-v2"
comptime HalfPtr = Pointer[Float16, MutUntrackedOrigin]


def cache_dir() -> String:
    var xdg = getenv("XDG_CACHE_HOME")
    return (xdg if xdg else getenv("HOME") + "/.cache") + "/tinybark"


def weights_file() -> String:
    return cache_dir() + "/" + NAME + ".safetensors"


@fieldwise_init
struct Entry(Copyable):
    """Where a tensor's fp16 data starts, in elements, and its shape with
    trailing axes folded into columns."""

    var offset: Int
    var rows: Int
    var cols: Int


struct Weights(Movable):
    # fp32 copies of the tensors whose names start with one of `fp32_prefixes`.
    var tensors: Dict[String, Tensor]
    var entries: Dict[String, Entry]
    var half: HalfPtr
    var has_half: Bool

    def __init__(out self, path: String, fp32_prefixes: List[String] = [""]) raises:
        """Reads the whole data block once."""
        var json = Python.import_module("json")
        var file = open(path, "r")
        var size_bytes = file.read_bytes(8)
        var header_size = 0
        for i in range(8):
            header_size |= Int(size_bytes[i]) << (8 * i)
        var header = json.loads(file.read(header_size))
        var total = 0
        self.entries = {}
        for name in header:
            var key = String(py=name)
            if key == "__metadata__":
                continue
            var info = header[name]
            var shape = info["shape"]
            var rows = 1
            var cols = 1
            for i in range(len(shape)):
                var n = Int(py=shape[i])
                if i == 0 and len(shape) > 1:
                    rows = n
                else:
                    cols *= n
            self.entries[key] = Entry(Int(py=info["data_offsets"][0]) // 2, rows, cols)
            total = max(total, Int(py=info["data_offsets"][1]))
        var data = unsafe_alloc[UInt8](total, alignment=64)
        var read = file.read(Span(unsafe_ptr=data, length=total))
        if read != total:
            raise Error("short read of ", path)
        file.close()
        self.half = data.unsafe_bitcast[Float16]()
        self.has_half = True

        self.tensors = {}
        for item in self.entries.items():
            var wanted = False
            for prefix in fp32_prefixes:
                wanted = wanted or item.key.startswith(prefix)
            if not wanted:
                continue
            ref e = item.value
            var t = Tensor(e.rows, e.cols)
            var src = self.half.unsafe_offset(e.offset)
            var n = e.rows * e.cols
            var j = 0
            while j + W <= n:
                t.ptr.unsafe_store(j, src.unsafe_load[width=W](j).cast[DType.float32]())
                j += W
            while j < n:
                t.ptr[unsafe_offset=j] = src[unsafe_offset=j].cast[DType.float32]()
                j += 1
            self.tensors[item.key] = t^

    def __deinit__(deinit self):
        if self.has_half:
            self.half.unsafe_bitcast[UInt8]().unsafe_free()

    def release_fp16(mut self):
        """Frees the fp16 data once it's on the GPU; the fp32 tensors stay."""
        if self.has_half:
            self.half.unsafe_bitcast[UInt8]().unsafe_free()
            self.has_half = False

    def __getitem__(ref self, name: String) raises -> ref[origin_of(self.tensors)._get_owned_interior["value"]] Tensor:
        return self.tensors[name]

    def fp16(self, name: String) raises -> Tuple[HalfPtr, Entry]:
        """A tensor's fp16 data, valid while self lives, and its entry."""
        if not self.has_half:
            raise Error("fp16 data already released")
        var e = self.entries[name].copy()
        return (self.half.unsafe_offset(e.offset), e^)
