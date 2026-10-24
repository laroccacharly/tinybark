"""Bark as a server for tinybark.mojo: loads once, prints "ready", then for each
line on stdin, the path of raw float32 16 kHz mono samples, prints "tokens [...]".

The encoder runs in fp16 on the GPU; the decoder and joint run on the CPU in
fp32, as in tinybark's ONNX Runtime setup. Send one line at a time and wait for
its reply: input() drops lines that arrive in the same read."""

from std.sys import has_accelerator

from max.gpu.host import DeviceContext

from bark.audio import Features
from bark.gpu import GpuEncoder
from bark.model import Parakeet
from bark.weights import Weights, weights_file


def read_f32(path: String) raises -> List[Float32]:
    var data = open(path, "r").read_bytes()
    var src = data.unsafe_ptr().unsafe_bitcast[Float32]()
    var out = List[Float32](capacity=len(data) // 4)
    for i in range(len(data) // 4):
        out.append(src[unsafe_offset=i])
    return out^


def main() raises:
    if not has_accelerator():
        raise Error("no GPU found")
    var weights = Weights(weights_file(), ["decoder.", "joint."])
    var encoder = GpuEncoder(DeviceContext(), weights)
    weights.release_fp16()
    var model = Parakeet(weights^)
    var features = Features()
    print("ready", encoder.ctx.name(), flush=True)

    while True:
        var path: String
        try:
            path = input()
        except:
            return
        var feats = features(read_f32(path))
        var tokens = List[Int]()
        if feats.rows >= 2:
            tokens = model.decode(encoder.encode(feats), feats.rows)
        print("tokens", tokens, flush=True)
