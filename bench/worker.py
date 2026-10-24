"""One backend in a fresh process: python -m bench.worker <backend> <gpu|cpu> <runs> <events.jsonl>

Appends one JSON line per event as it happens (load, then each run), so a crash
mid-benchmark keeps what was measured before it."""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

CLIPS = Path(__file__).parents[1] / "clips"
NAMES = ["quilter", "quilter3"]

# transformers' generate() notice, on every call.
warnings.filterwarnings("ignore", "Using the model-agnostic default `max_length`")


def clip(name: str) -> tuple[np.ndarray, dict]:
    samples = np.frombuffer((CLIPS / f"{name}.pcm").read_bytes(), dtype="<i2").astype(np.float32) / 32768
    return samples, json.loads((CLIPS / f"{name}.json").read_text())


def main(backend: str, device: str, runs: int, out: Path) -> None:
    def emit(**event) -> None:
        with out.open("a") as file:
            file.write(json.dumps(event) + "\n")

    start = time.perf_counter()
    from tinybark.backends import BACKENDS

    bark = BACKENDS[backend].load(device)
    emit(event="load", seconds=time.perf_counter() - start, device=bark.device)

    for name in NAMES:
        samples, reference = clip(name)
        for run in range(runs):
            start = time.perf_counter()
            tokens = bark.tokens(samples)
            seconds = time.perf_counter() - start
            emit(event="run", clip=name, run=run, seconds=seconds, tokens=tokens, match=tokens == reference["tokens"])
            print(f"{backend} ({device}): {name} run {run + 1}: {seconds:.3f} s{'' if tokens == reference['tokens'] else ', tokens differ from the reference'}", file=sys.stderr, flush=True)
    if hasattr(bark, "close"):
        bark.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]))
