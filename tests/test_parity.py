"""tinybark, on tinygrad and on ONNX Runtime, must transcribe like the PyTorch
model it replaces (set up as stt's bark.py runs it), token for token. Prints the
load time, the first (cold) call and the warm calls of each.

    uv run pytest -k onnx  # PyTorch against ONNX Runtime only"""

import gc
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from tinygrad import Device
from transformers import AutoProcessor

import tinybark
from tinybark import export, ort, weights

RUNS = 3
PCM = (Path(__file__).parent / "fixtures/quilter.pcm").read_bytes()
SAMPLES = np.frombuffer(PCM, dtype="<i2").astype(np.float32) / 32768


def timed(fn, runs: int) -> tuple[list[float], object]:
    times, out = [], None
    for _ in range(runs):
        start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - start)
    return times, out


def pytorch() -> dict:
    # bark.py's settings: bf16 where the CPU has AVX512 BF16, else fp32.
    bf16 = getattr(torch.cpu, "_is_avx512_bf16_supported", lambda: False)()
    start = time.perf_counter()
    model = export.torch_model(torch.bfloat16 if bf16 else torch.float32)
    processor = AutoProcessor.from_pretrained(weights.TOKENIZER_REPO)
    load = time.perf_counter() - start

    def run():
        inputs = processor([SAMPLES], sampling_rate=16_000, return_tensors="pt", padding=True)
        dtype = next(model.parameters()).dtype
        inputs = {k: v.to(dtype) if v.dtype.is_floating_point else v for k, v in inputs.items()}
        with torch.inference_mode():
            sequence = model.generate(**inputs).sequences[0].tolist()
        return sequence, processor.batch_decode([sequence], skip_special_tokens=True)[0].strip()

    first, _ = timed(run, 1)
    times, (sequence, text) = timed(run, RUNS)
    skip = {model.config.blank_token_id, model.config.pad_token_id}
    tokens = [t for t in sequence[1:] if t not in skip]
    del model
    gc.collect()
    return {"name": f"pytorch ({'bf16' if bf16 else 'fp32'}, CPU)", "load": load, "first": first[0], "times": times, "tokens": tokens, "text": text}


def tinygrad() -> dict:
    start = time.perf_counter()
    # Warms up only the clip's padded length, not every length up to 30 s.
    bark = tinybark.load(max_seconds=len(SAMPLES) / 16_000)
    load = time.perf_counter() - start
    first, _ = timed(lambda: bark.tokens(SAMPLES), 1)
    times, tokens = timed(lambda: bark.tokens(SAMPLES), RUNS)
    return {"name": f"tinygrad (fp16, {Device.DEFAULT})", "load": load, "first": first[0], "times": times, "tokens": tokens, "text": bark.tokenizer.decode(tokens)}


def onnx() -> dict:
    ort.load()  # Exports the graphs on first use, outside the timing.
    start = time.perf_counter()
    bark = ort.load()
    load = time.perf_counter() - start
    first, _ = timed(lambda: bark.tokens(SAMPLES), 1)
    times, tokens = timed(lambda: bark.tokens(SAMPLES), RUNS)
    return {"name": "onnxruntime (fp32, CPU)", "load": load, "first": first[0], "times": times, "tokens": tokens, "text": bark.tokenizer.decode(tokens)}


def report(results: list[dict]) -> None:
    seconds = len(SAMPLES) / 16_000
    print(f"\n{seconds:.2f} s of audio, median of {RUNS} warm runs")
    print(f"| {'':24} | {'load s':>7} | {'first s':>7} | {'median s':>8} | {'x realtime':>10} |")
    for r in results:
        median = statistics.median(r["times"])
        print(f"| {r['name']:24} | {r['load']:7.2f} | {r['first']:7.2f} | {median:8.3f} | {seconds / median:10.0f} |")
    for r in results:
        print(f"{r['name']}: {r['text']!r}")


@pytest.fixture(scope="module")
def reference() -> dict:
    return pytorch()


@pytest.mark.parametrize("run", [tinygrad, onnx])
def test_matches_pytorch(reference, run):
    ours = run()
    report([reference, ours])
    assert ours["tokens"] == reference["tokens"]
    assert ours["text"] == reference["text"]
