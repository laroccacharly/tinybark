# Learnings, 2026-10-24

Ryzen 5 7640HS (6 cores, 12 threads, AVX512 BF16), Radeon 760M iGPU (gfx1103).
Clips: `quilter` (5.86 s) and `quilter3` (17.6 s). Every backend and device
below produced the reference tokens.

## Results, warm best (seconds)

| backend | GPU 5.86 s | GPU 17.6 s | CPU 5.86 s | CPU 17.6 s |
|---|---|---|---|---|
| pytorch | 0.153 | 0.340 | 0.205 | 0.477 |
| pytorch-compile | 0.144 | 0.348 | 0.207 | 0.605 |
| tinygrad | 0.124 | 0.464 | 11.9 | 36.0 |
| onnx | 0.153 | 0.377 | 0.311 | 0.892 |
| mojo | 0.128 | 0.300 | 0.320 | 0.786 |

"GPU" for onnx and mojo means the encoder on the GPU, and the decoder and
joint on the CPU. pytorch and tinygrad run everything on the GPU.

## GPU

- Mojo is the fastest, loads in 2–3 s and has almost no first-call cost.
- PyTorch eager pays 2.6–6.3 s on the first call at a new length. With the
  decoder and joint moved to the CPU (fp32), that cost drops to about 0.7 s,
  but warm runs get 1.75x slower (0.27 s vs 0.15 s). So the CPU decoder is
  not why Mojo is fast. Its compiled decoding loop probably avoids PyTorch's
  per-op overhead (not profiled).
- `torch.compile` gives no warm speedup over eager and costs 28–40 s of
  compiling per new length.
- tinygrad's second run at a length is still slow (0.9–1.2 s); only the third
  is warm. Use "warm best", not the warm mean, to compare it.
- On the CPU, the GPU is only 1.3–1.6x faster than PyTorch bf16 on the long clip.

## CPU

- PyTorch in bf16 is the fastest, most likely because this CPU runs bf16
  natively (AVX512 BF16).
- `torch.compile` on the CPU costs 43 s and 136 s of compiling and is no
  faster than eager.
- tinygrad on the CPU is about 60x slower than PyTorch. Breakdown on the
  5.86 s clip: about 7.2 s in the encoder and about 4.7 s in 34 decoder steps.
  The top kernels are the encoder matmuls: 3.4 s in the subsampling linear
  layer, and 2.2 s each in the feed-forward up and down projections.
  Things tried (warm, 5.86 s clip):

  | tinygrad CPU setup | warm |
  |---|---|
  | fp16, Clang (default) | 11.9 s |
  | fp16, LLVM (`DEV=CPU:LLVM`) | 21.2 s |
  | fp32, Clang | 8.5 s |
  | fp32, LLVM | 7.7 s |

  fp32 helps a little: there's no native fp16 math on this CPU. But even the best
  setup is about 40x off PyTorch. tinygrad 0.14's CPU renderers have
  `global_max = (1, 0, 0)`, and `CPUProgram` ignores `global_size`, so each
  kernel seems to run on a single thread. That is the likely main cause, not
  confirmed. Not fixed.

## Mojo in bf16 on the CPU (probe, not implemented)

Could Mojo on the CPU catch up with PyTorch by running in bf16, as PyTorch
does? Timed the encoder's linear layers alone (×24 layers), at the frame counts
of both clips, 6 threads:

| encoder linears | 5.86 s | 17.6 s |
|---|---|---|
| MAX `linalg.matmul`, fp32 | 145 ms | 384 ms |
| MAX `linalg.matmul`, bf16 in, fp32 out | 144 ms | 395 ms |
| PyTorch, fp32 | 201 ms | 454 ms |
| PyTorch, bf16 | 78 ms | 171 ms |

- The matmuls are about half of Mojo's CPU time (0.32 s and 0.79 s end to end).
- MAX's fp32 matmul already runs at 600–640 GFLOP/s, near this CPU's fp32
  peak. Its bf16 matmul is no faster (0.91–1.10x); it doesn't seem to use
  AVX512 BF16.
- PyTorch's bf16 reaches 1.0–1.6 TFLOP/s, 2.5x MAX's fp32.
- Mojo can emit `vdpbf16ps` through
  `llvm_intrinsic["llvm.x86.avx512bf16.dpbf16ps.512"]`: about 300 GFLOP/s on
  one core, so about 1.8 TFLOP/s on six.
- So the gain needs our own bf16 matmul kernel: weights converted once at load
  into the layout `vdpbf16ps` reads (pairs along k), tiled and parallel over
  cores. At PyTorch's speed that would take Mojo to about 0.25 s and 0.57 s,
  still behind PyTorch (0.205 s and 0.477 s): the other half of the time is
  attention, the convolution module and the decoder, partly unvectorized
  (e.g. the GLU loop in `conv`). Profile that half before writing the kernel.
- fp16 isn't an option on this CPU: no AVX512 FP16, only F16C conversions, as
  tinygrad's fp16 vs fp32 CPU times show. For ONNX Runtime (not tried): its x86
  CPU kernels likely have little fp16 and almost no bf16 support; int8 dynamic
  quantization (AVX512 VNNI) is the more promising direction.

## Running the benchmark

- Run one GPU backend at a time. Running two at once crashed Mojo inside
  the HSA runtime (SIGABRT).
- About 1 GPU start in 10 hangs the 760M (`MES failed to respond`, then a GPU
  reset). One hang took down Hyprland. Rerunning works.
- ONNX load time depends on a warm cache: 11.4 s the first time, 2.7 s after.

## Native number formats on this machine

Any format can be stored anywhere; what matters is whether the hardware does
the math in it, or converts to a wider format first (less memory, no speedup).
CPU flags from `/proc/cpuinfo`; GPU and NPU support from their specs, not
tested here except fp16 on the GPU and bf16 (`vdpbf16ps`) on the CPU.

| format | CPU (Zen 4) | GPU (760M, RDNA3, gfx1103) | NPU (XDNA 1, Phoenix) |
|---|---|---|---|
| fp32 | native, AVX-512 FMA | native | not checked |
| fp16 | conversion only (`f16c`) | native, WMMA, ~2x fp32 | not checked |
| bf16 | native (`avx512_bf16`), ~2x fp32 | native, WMMA | yes (Ryzen AI's bf16 models) |
| fp8 | no | no (RDNA4 and up) | not checked |
| int8 | native (`avx512_vnni`) | native, WMMA and dot instructions | yes (Ryzen AI's int8 models), ~10 TOPS |
| int4 | no, unpack to int8 | native, WMMA | not checked |

- fp16 has 5 exponent bits and 10 of mantissa (more precise, max 65504); bf16
  has fp32's 8 exponent bits and 7 of mantissa (fp32's range, less precise).
  bf16 is fp32 with the low 16 bits cut off.
- The NPU is there and seen by the kernel: `amdxdna` is loaded and
  `/dev/accel/accel0` exists.

## The NPU on Linux

- On this chip (Phoenix, XDNA 1) the NPU is better supported on Windows.
  AMD's Ryzen AI Software 1.8 supports Phoenix, but its Linux flow (ONNX
  Runtime with the NPU execution provider, int8 and bf16 models) starts with
  Strix Point (XDNA 2). FastFlowLM, which runs LLMs on the NPU on Linux, also
  needs XDNA 2.
- On Linux, Phoenix has the in-tree `amdxdna` driver (kernel 6.14 and up) and
  IRON / MLIR-AIE, AMD's low-level toolkit for writing NPU kernels yourself
  (device `npu1`). Running Bark there means writing its kernels in IRON, not
  loading an ONNX graph.

## Integers or floats

Quantization is heading to fewer bits, but not simply to integers:

- Hardware is moving to low-bit floats with block scales ("microscaling"):
  FP8, MXFP4 (one shared power-of-two scale per 32 values) and NVFP4 (one
  FP8 scale per 16 values, about 4.5 bits per value, NVIDIA Blackwell).
- Small floats cope better with outliers in activations than plain integers.
  With small blocks the gap closes: a 2025 study ("INT v.s. FP", arXiv
  2510.25602) finds MXINT8 beats MXFP8 in accuracy and in chip area and
  energy, while at 4 bits floats (MXFP4, NVFP4) usually win unless a Hadamard
  rotation spreads the outliers first.
- The common thread is the shared per-block scale more than int vs float.
  Edge NPUs, this one included, are still mostly int8.
