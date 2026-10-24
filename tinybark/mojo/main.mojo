"""Bark as a server for tinybark.mojo: main <gpu|cpu> loads once, prints "ready",
then for each line on stdin, the path of raw float32 16 kHz mono samples, prints
"tokens [...]".

On "gpu", the encoder runs in fp16 on the GPU and the decoder and joint on the
CPU in fp32, as in tinybark's ONNX Runtime setup. On "cpu", everything runs on
the CPU in fp32. Send one line at a time and wait for
its reply: input() drops lines that arrive in the same read."""

from std.sys import argv, has_accelerator

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


def next_path(mut path: String) -> Bool:
    """The next request's samples path, or False once stdin is closed."""
    try:
        path = input()
        return True
    except:
        return False


def main() raises:
    var args = argv()
    var device = String(args[1]) if len(args) > 1 else String("gpu")
    var features = Features()
    var path = String()
    if device == "cpu":
        var model = Parakeet(Weights(weights_file()))
        print("ready CPU", flush=True)
        while next_path(path):
            print("tokens", model.generate(features(read_f32(path))), flush=True)
        return
    if device != "gpu":
        raise Error("device must be gpu or cpu, got " + device)

    if not has_accelerator():
        raise Error("no GPU found")
    var weights = Weights(weights_file(), ["decoder.", "joint."])
    var encoder = GpuEncoder(DeviceContext(), weights)
    weights.release_fp16()
    var model = Parakeet(weights^)
    print("ready", encoder.ctx.name(), flush=True)
    while next_path(path):
        var feats = features(read_f32(path))
        var tokens = List[Int]()
        if feats.rows >= 2:
            tokens = model.decode(encoder.encode(feats), feats.rows)
        print("tokens", tokens, flush=True)
