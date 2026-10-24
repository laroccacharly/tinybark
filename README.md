# tinybark

Benchmarks of [Underdog Bark 3.0 Clean](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Clean) speech-to-text (a Parakeet TDT 0.6B fine-tune) on four GPU backends, one folder each under `tinybark/`: [PyTorch](https://pytorch.org) (eager and `torch.compile`), [tinygrad](https://github.com/tinygrad/tinygrad), [ONNX Runtime](https://onnxruntime.ai) with WebGPU, and [Mojo](https://www.modular.com/mojo). Each one is also usable as a library.

## Requirements

- Linux x86_64, Python 3.14 and [uv](https://docs.astral.sh/uv/)
- An AMD GPU (tested on a Radeon 760M, gfx1103) with ROCm's HIP runtime (`hip-runtime-amd` and `comgr` on Arch) for PyTorch, tinygrad and Mojo, and Vulkan for ONNX Runtime
- About 20 GB of disk: PyTorch for ROCm alone is 15 GB

On the 760M, about 1 start in 10 hangs the GPU (a page fault in KFD's trap handler, then a GPU reset). A possible fix, untested: boot with `amdgpu.cwsr_enable=0`, which disables compute preemption system-wide.

## Install

```sh
uv sync
```

The model is downloaded and converted to `~/.cache/tinybark` on first use.
