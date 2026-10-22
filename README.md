# tinybark

[Underdog Bark 3.0 Clean](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Clean) speech-to-text (a Parakeet TDT 0.6B fine-tune) without PyTorch, on [tinygrad](https://github.com/tinygrad/tinygrad) or on [ONNX Runtime](https://onnxruntime.ai) with WebGPU. It transcribes token for token like the transformers model it ports.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- For ONNX Runtime: a Vulkan GPU (tested on a Radeon 760M with RADV), or the CPU. Exporting the ONNX graphs on first use needs the dev dependencies (`uv sync` installs them)
- For tinygrad: a GPU it supports (tested on a Radeon 760M through its `AMD` backend, which needs ROCm's `comgr`), or its CPU backend

On the 760M, about 1 start in 10 hangs the GPU (a page fault in KFD's trap handler, then a GPU reset). A possible fix, untested: boot with `amdgpu.cwsr_enable=0`, which disables compute preemption system-wide.

## Install

```sh
uv sync
```

The model is downloaded and converted to `~/.cache/tinybark` on first use.
