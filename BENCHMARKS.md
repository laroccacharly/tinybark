# tinybark benchmarks

2026-10-21. Ryzen 5 7640HS, Radeon 760M iGPU (gfx1103), 14.7 GB shared RAM.
tinygrad `45fc5ae73` (`DEV=AMD`), torch 2.14.1+cpu, transformers 5.19.0.
Test clip: `tests/fixtures/quilter.pcm`, 5.86 s.

## Summary

- tinybark transcribes token for token like the PyTorch model it ports, on the
  test clip and on three more clips (3 s, 29 s, 2 s of silence).
- Once warm, tinybark is about 2x faster: 0.12–0.14 s per transcription against
  0.21–0.23 s for PyTorch (bf16 on the CPU, as stt's `bark.py` runs it).
- PyTorch loads slowly (about 30 s) but has almost no warm-up afterwards.
  tinybark loads in 6–7 s but pays about 5.6 s the first time it sees each
  padded audio length (a multiple of 2.56 s).
- That per-length cost is tinygrad's Python scheduling the work, not kernel
  compilation: with an empty kernel cache the first call took 4.91 s against
  4.80 s with a warm one.

## Speed, warm

| | 3 s | 5.86 s | 29 s | 2 s silence |
|---|---|---|---|---|
| PyTorch (bf16, CPU) | 0.27 s | 0.21–0.23 s | 1.29 s | 0.12 s |
| tinybark (fp16, 760M) | 0.10 s | 0.12–0.14 s | 0.73 s | 0.08 s |

On the 5.86 s clip, a warm call breaks down as 0.08 s for the encoder, 0.04 s for
34 decoder steps and 0.02 s for the rest.

## Loading: PyTorch

stt's `bark.py` repeats all of this on every server start; nothing is cached.

| Phase | Time |
|---|---|
| import torch | 1.0 s |
| import transformers | 3.0–3.4 s |
| Decompress `.bark` (ANS entropy decoding) | 15.0 s |
| Build the model with random weights | 4.3–5.9 s |
| `load_state_dict` (fp32) | 1.2–2.8 s |
| Cast to bf16, gc, `malloc_trim` | 3.3–3.8 s |
| Processor and tokenizer | 1.2 s |
| **Loaded** | **about 30 s** |
| 1st transcription | 0.4–0.5 s |
| Later transcriptions | 0.21 s |
| 1st transcription at a new length (11.7 s) | 0.35 s |

## Loading: tinybark

Once, when `~/.cache/tinybark` is empty (about 23 s): decompress `.bark` 15.9 s,
fuse and reshape the weights 2.2 s, write the 1.25 GB fp16 safetensors 4.8 s.

Every start (about 6–7 s):

| Phase | Time |
|---|---|
| import tinygrad, open the GPU | 0.25–0.45 s |
| Read the safetensors header | 0.15 s |
| Disk to GPU, one 1.25 GB read | 1.75–2.05 s |
| Split into 577 GPU buffers | 3.1–3.8 s |
| Tokenizer | 0.5–0.65 s |

Every new padded length:

| Call | 5.86 s clip | 11.7 s clip | |
|---|---|---|---|
| Relative position keys | 0.3 s | | once per length |
| 1st | 4.8 s (encoder 3.7, decoder step 0.55) | 6.1 s | trace, schedule, compile |
| 2nd | 0.8 s | 0.9 s | TinyJit capture |
| 3rd and later | 0.14 s | 0.22 s | TinyJit replay |

Profile of a first call: about 75% scheduling (`create_linear_with_vars`), about
10% code generation (`to_program`), 0.11 s compiling to machine code
(`compile_cached`). tinygrad's kernel cache (`~/.cache/tinygrad/cache.db`)
already stores compiled kernels but not captured JITs, so it saves almost nothing
here.

## What made tinybark fast

From 0.56 s to 0.12 s per transcription on the 5.86 s clip:

- Decoder state (LSTM `h`/`c`) and the token kept on the GPU, the token passed as a
  bound `Variable`: building inputs from NumPy cost more than a decoder step.
- Encoder masks built inside the JIT from bound length `Variable`s instead of
  uploaded from NumPy on each call.
- Two-stage argmax over the 8193 token logits (padded to 8256 with `-inf`); one
  reduction ran nearly serially. Its inner stage needs `contiguous()`.
- LSTM input and hidden weights fused into one matmul per layer; q/k/v fused;
  batch norm folded into the depthwise conv; the joint head split into token and
  duration heads.
- Relative position keys precomputed per length (24 projections that depend only
  on the length).
- fp16 subsampling convolutions.
- One padding size (multiples of 2.56 s) instead of power-of-two buckets: less
  wasted encoder work.

Tried and dropped:

- `BEAM=2`: encoder GPU time 126 ms to 105 ms, after 6 minutes of search.
- Weights as views into one big GPU buffer (loads in 2 s instead of 5 s): every
  transcription got slower, 0.18 s instead of 0.12 s, and the first call took 21 s.
- Loading through a NumPy memory map: 6 s, but 2.5 GB peak RSS.

## Decision: warm up in memory at start

`tinybark.load()` warms up every padded length up to 30 s, two calls each, and
keeps the captured JITs in memory: about 70 s for the 12 lengths, paid on every
start. Two ways around it were tried and dropped; both are kept in
`tinybark/attempts/`:

- Symbolic length (`symbolic.py`): one graph for every length up to 30.72 s, so
  one compile instead of 12. Warm calls were 4-10x slower (0.63 s instead of
  0.12-0.14 s on the 5.86 s clip, 7.5 s instead of 0.73 s on 29 s, nearly all in
  the encoder).
- Freezing the warm model to disk (`freeze.py`): pickled JITs restored on start.
  It needed workarounds for tinygrad internals, a 1.29 GB pickle, and wasn't
  reliable.

## GPU faults

About 1 fresh process in 10 hangs the GPU, with both the bucketed and the symbolic
model: a page fault on an instruction fetch at `0xffffffbf0000` (KFD's trap
handler), then MES fails to remove the queue and the driver resets the GPU. It
happens when a wave traps, as on preemption (CWSR), which MES uses to time-slice
the GPU with the desktop. KFD rejects compute queues without a context save area
(`EINVAL`), so tinygrad can't opt out; booting with `amdgpu.cwsr_enable=0`
disables preemption for the whole system.
