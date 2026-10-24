"""Benchmarks every backend on a 5.86 s and a 17.6 s clip, three runs each: the
first run is the first call at that clip's length (cold), the next two are warm.

    uv run python -m bench                 # build, then run every backend
    uv run python -m bench build           # downloads, ONNX export, Mojo build; no GPU
    uv run python -m bench run onnx mojo   # some backends only, already built

Each backend runs alone in a fresh process. Results go to results/<time>.json.
Tokens that differ from the reference (clips/*.json) are reported, not fatal."""

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

from tinybark.backends import BACKENDS

from .worker import NAMES, clip

ROOT = Path(__file__).parents[1]
# Loading plus six runs; a hung GPU otherwise blocks the whole benchmark.
TIMEOUT = 1200


def build(names: list[str]) -> None:
    for name in names:
        start = time.perf_counter()
        print(f"build {name}", file=sys.stderr, flush=True)
        BACKENDS[name].prepare()
        print(f"build {name}: {time.perf_counter() - start:.1f} s", file=sys.stderr, flush=True)


def run(name: str) -> dict:
    backend = BACKENDS[name]
    env = os.environ | backend.env
    scratch = Path(tempfile.mkdtemp(prefix=f"bench-{name}-"))
    # An empty inductor cache, so torch.compile's first call includes compiling.
    env["TORCHINDUCTOR_CACHE_DIR"] = str(scratch / "inductor")
    events = scratch / "events.jsonl"
    start = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, "-m", "bench.worker", name, events], cwd=ROOT, env=env, timeout=TIMEOUT)
        error = None if process.returncode == 0 else f"exited with {process.returncode}"
    except subprocess.TimeoutExpired:
        error = f"timed out after {TIMEOUT} s"
    result = {"backend": name, "description": backend.description, "env": backend.env, "wall_seconds": time.perf_counter() - start, "error": error, "load_seconds": None, "device": None, "clips": {}}
    for line in events.read_text().splitlines() if events.exists() else []:
        event = json.loads(line)
        if event["event"] == "load":
            result["load_seconds"], result["device"] = event["seconds"], event["device"]
        else:
            runs = result["clips"].setdefault(event["clip"], [])
            runs.append({k: event[k] for k in ("seconds", "match", "tokens")})
    shutil.rmtree(scratch, ignore_errors=True)
    return result


def versions() -> dict:
    out = {"python": platform.python_version()}
    for package in ("torch", "transformers", "tinygrad", "onnxruntime-webgpu", "mojo", "max"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return out


def machine() -> dict:
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor())
    return {"cpu": cpu, "kernel": platform.release()}


def report(results: list[dict]) -> str:
    rows = ["| backend | load s | clip | run 1 (cold) s | run 2 s | run 3 s | warm mean s | tokens |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        load = f"{r['load_seconds']:.2f}" if r["load_seconds"] is not None else "-"
        if not r["clips"]:
            rows.append(f"| {r['backend']} | {load} | | | | | | {r['error'] or ''} |")
        for name, runs in r["clips"].items():
            times = [f"{x['seconds']:.3f}" for x in runs] + ["-"] * (3 - len(runs))
            warm = [x["seconds"] for x in runs[1:]]
            match = "match" if all(x["match"] for x in runs) else "differ"
            rows.append(f"| {r['backend']} | {load} | {name} | {' | '.join(times)} | {f'{statistics.mean(warm):.3f}' if warm else '-'} | {match} |")
            load = ""
        if r["error"] and r["clips"]:
            rows.append(f"| {r['backend']} | | | | | | | {r['error']} |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bench")
    parser.add_argument("command", nargs="?", choices=["all", "build", "run"], default="all")
    parser.add_argument("backends", nargs="*", help=", ".join(BACKENDS))
    args = parser.parse_args()
    names = args.backends or list(BACKENDS)
    if unknown := set(names) - set(BACKENDS):
        parser.error(f"unknown backends {sorted(unknown)}, choose from {', '.join(BACKENDS)}")
    if args.command in ("all", "build"):
        build(names)
    if args.command == "build":
        return

    results = []
    for name in names:
        print(f"\n== {name}: {BACKENDS[name].description}", file=sys.stderr, flush=True)
        results.append(run(name))
        if results[-1]["error"]:
            print(f"{name}: {results[-1]['error']}", file=sys.stderr, flush=True)

    clips = {name: {"seconds": len(clip(name)[0]) / 16_000, "reference_tokens": len(clip(name)[1]["tokens"])} for name in NAMES}
    stamp = datetime.now().astimezone()
    out = ROOT / "results" / f"{stamp:%Y-%m-%dT%H-%M-%S}.json"
    out.parent.mkdir(exist_ok=True)
    data = {"time": stamp.isoformat(timespec="seconds"), "machine": machine(), "versions": versions(), "clips": clips, "runs_per_clip": 3, "results": results}
    out.write_text(json.dumps(data, indent=2) + "\n")
    print("\n" + report(results))
    print(f"\n{out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
