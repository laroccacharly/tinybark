"""Benchmarks backends on a 5.86 s and a 17.6 s clip, on the GPU or the CPU: the
first run is the first call at that clip's length (cold), the rest are warm.

    uv run python -m bench                   # every backend on the GPU
    uv run python -m bench onnx mojo -d cpu  # some backends, on the CPU
    uv run python -m bench -d all -n 5       # both devices, five runs per clip
    uv run python -m bench --build-only      # downloads, ONNX export, Mojo build; no GPU
    uv run python -m bench --list

Each backend runs alone in a fresh process, one device at a time. Results go to
results/<time>.json. Tokens that differ from the reference (clips/*.json) are
reported, not fatal."""

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

from tinybark.backends import BACKENDS, DEVICES

from .worker import NAMES, clip

ROOT = Path(__file__).parents[1]
# Per backend and device: a hung GPU otherwise blocks the whole benchmark.
TIMEOUT = 1800


def build(names: list[str]) -> None:
    for name in names:
        start = time.perf_counter()
        print(f"build {name}", file=sys.stderr, flush=True)
        BACKENDS[name].prepare()
        print(f"build {name}: {time.perf_counter() - start:.1f} s", file=sys.stderr, flush=True)


def run(name: str, device: str, runs: int) -> dict:
    backend = BACKENDS[name]
    env = os.environ | backend.env(device)
    scratch = Path(tempfile.mkdtemp(prefix=f"bench-{name}-{device}-"))
    # An empty inductor cache, so torch.compile's first call includes compiling.
    env["TORCHINDUCTOR_CACHE_DIR"] = str(scratch / "inductor")
    events = scratch / "events.jsonl"
    start = time.perf_counter()
    try:
        command = [sys.executable, "-m", "bench.worker", name, device, str(runs), events]
        process = subprocess.run(command, cwd=ROOT, env=env, timeout=TIMEOUT)
        error = None if process.returncode == 0 else f"exited with {process.returncode}"
    except subprocess.TimeoutExpired:
        error = f"timed out after {TIMEOUT} s"
    result = {
        "backend": name,
        "device": device,
        "description": backend.descriptions[device],
        "env": backend.env(device),
        "wall_seconds": time.perf_counter() - start,
        "error": error,
        "load_seconds": None,
        "hardware": None,
        "clips": {},
    }
    for line in events.read_text().splitlines() if events.exists() else []:
        event = json.loads(line)
        if event["event"] == "load":
            result["load_seconds"], result["hardware"] = event["seconds"], event["device"]
        else:
            result["clips"].setdefault(event["clip"], []).append({k: event[k] for k in ("seconds", "match", "tokens")})
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
    rows = ["| backend | device | load s | clip | cold s | warm mean s | warm best s | tokens |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        head = f"| {r['backend']} | {r['device']} | {f'{r["load_seconds"]:.2f}' if r['load_seconds'] is not None else '-'}"
        if not r["clips"]:
            rows.append(f"{head} | | | | | {r['error'] or ''} |")
        for name, runs in r["clips"].items():
            warm = [x["seconds"] for x in runs[1:]]
            mean, best = (f"{statistics.mean(warm):.3f}", f"{min(warm):.3f}") if warm else ("-", "-")
            match = "match" if all(x["match"] for x in runs) else "differ"
            rows.append(f"{head} | {name} | {runs[0]['seconds']:.3f} | {mean} | {best} | {match} |")
            head = f"| {r['backend']} | {r['device']} |"
        if r["error"] and r["clips"]:
            rows.append(f"| {r['backend']} | {r['device']} | | | | | | {r['error']} |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bench", description="Benchmark tinybark's backends.")
    parser.add_argument("backends", nargs="*", metavar="backend", help=f"any of {', '.join(BACKENDS)} (default: all)")
    parser.add_argument("-d", "--device", choices=[*DEVICES, "all"], default="gpu", help="where to run (default: gpu)")
    parser.add_argument("-n", "--runs", type=int, default=3, help="runs per clip, the first one cold (default: 3)")
    steps = parser.add_mutually_exclusive_group()
    steps.add_argument("--build-only", action="store_true", help="prepare the backends, don't run them")
    steps.add_argument("--no-build", action="store_true", help="skip preparing, the backends are already built")
    parser.add_argument("--list", action="store_true", help="list the backends and what runs where")
    args = parser.parse_args()
    names = args.backends or list(BACKENDS)
    if unknown := [n for n in names if n not in BACKENDS]:
        parser.error(f"unknown backend {', '.join(unknown)}; choose from {', '.join(BACKENDS)}")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    devices = DEVICES if args.device == "all" else (args.device,)

    if args.list:
        for name in names:
            for device, description in BACKENDS[name].descriptions.items():
                print(f"{name:16} {device:4} {description}")
        return
    if not args.no_build:
        build(names)
    if args.build_only:
        return

    results = []
    for device in devices:
        for name in names:
            if device not in BACKENDS[name].descriptions:
                continue
            print(f"\n== {name} ({device}): {BACKENDS[name].descriptions[device]}", file=sys.stderr, flush=True)
            results.append(run(name, device, args.runs))
            if results[-1]["error"]:
                print(f"{name} ({device}): {results[-1]['error']}", file=sys.stderr, flush=True)

    clips = {name: {"seconds": len(clip(name)[0]) / 16_000, "reference_tokens": len(clip(name)[1]["tokens"])} for name in NAMES}
    stamp = datetime.now().astimezone()
    out = ROOT / "results" / f"{stamp:%Y-%m-%dT%H-%M-%S}.json"
    out.parent.mkdir(exist_ok=True)
    data = {"time": stamp.isoformat(timespec="seconds"), "machine": machine(), "versions": versions(), "clips": clips, "runs_per_clip": args.runs, "results": results}
    out.write_text(json.dumps(data, indent=2) + "\n")
    print("\n" + report(results))
    print(f"\n{out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
