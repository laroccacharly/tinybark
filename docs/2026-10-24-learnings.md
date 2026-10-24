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

## Running the benchmark

- Run one GPU backend at a time. Running two at once crashed Mojo inside
  the HSA runtime (SIGABRT).
- About 1 GPU start in 10 hangs the 760M (`MES failed to respond`, then a GPU
  reset). One hang took down Hyprland. Rerunning works.
- ONNX load time depends on a warm cache: 11.4 s the first time, 2.7 s after.
