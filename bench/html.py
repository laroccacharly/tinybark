"""Writes results/index.html: an interactive page with every run's time, per backend,
device and clip, and each backend's lines of code.

    uv run python -m bench.html

Takes the latest result per backend and device from results/*.json. Results from
before the -d flag have the hardware's name as their device: those ran on the GPU."""

import json
from pathlib import Path

from tinybark.backends import BACKENDS, DEVICES

ROOT = Path(__file__).parents[1]
# Failed experiments, not part of a backend.
SKIP = {"attempts", "__pycache__"}
# Package under tinybark/ per backend, where it isn't the backend's name.
FOLDERS = {"onnx": "ort"}


def latest() -> dict:
    files = sorted((ROOT / "results").glob("*.json"))
    picked, clips, machine = {}, {}, {}
    for file in files:
        data = json.loads(file.read_text())
        clips |= data["clips"]
        machine = data["machine"]
        for r in data["results"]:
            device = r["device"] if r["device"] in DEVICES else "gpu"
            if r["clips"]:
                picked[(r["backend"], device)] = {
                    "backend": r["backend"],
                    "device": device,
                    "description": r["description"],
                    "file": file.name,
                    "load": r["load_seconds"],
                    "clips": {name: [x["seconds"] for x in runs] for name, runs in r["clips"].items()},
                    "match": all(x["match"] for runs in r["clips"].values() for x in runs),
                }
    return {"machine": machine, "clips": clips, "results": list(picked.values())}


def lines_of_code() -> list[dict]:
    out = []
    for name in dict.fromkeys(name.split("-")[0] for name in BACKENDS):
        folder = ROOT / "tinybark" / FOLDERS.get(name, name)
        files = [f for f in sorted(folder.rglob("*")) if f.suffix in (".py", ".mojo") and not SKIP & set(f.parts)]
        count = {f.relative_to(ROOT).as_posix(): sum(1 for line in f.read_text().splitlines() if line.strip()) for f in files}
        out.append({"backend": name, "lines": sum(count.values()), "files": count})
    return out


def main() -> None:
    data = latest() | {"backends": list(BACKENDS), "loc": lines_of_code()}
    out = ROOT / "results" / "index.html"
    out.write_text(PAGE.replace("/*DATA*/null", json.dumps(data)))
    print(out.relative_to(ROOT))


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>tinybark benchmarks</title>
<style>
  :root { --ink: #1d1d1f; --muted: #6e6e73; --line: #d9d9de; --bg: #fbfbfa; }
  body { font: 14px/1.45 ui-sans-serif, system-ui, sans-serif; color: var(--ink); background: var(--bg); margin: 0; padding: 32px 40px; max-width: 1100px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 40px 0 8px; }
  .muted { color: var(--muted); }
  .controls { display: flex; flex-wrap: wrap; gap: 24px; margin: 20px 0 8px; }
  fieldset { border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; margin: 0; }
  legend { color: var(--muted); font-size: 12px; padding: 0 4px; }
  label { margin-right: 10px; cursor: pointer; white-space: nowrap; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
  svg { display: block; overflow: visible; }
  svg text { font-size: 12px; fill: var(--ink); }
  .axis line, .axis path { stroke: var(--ink); }
  .grid line { stroke: var(--line); }
  rect.bar:hover { opacity: .75; }
  #tip { position: fixed; pointer-events: none; background: var(--ink); color: #fff; padding: 6px 8px; border-radius: 6px; font-size: 12px; display: none; max-width: 340px; }
</style>
</head>
<body>
<h1>tinybark benchmarks</h1>
<div class="muted" id="machine"></div>

<h2>Runtime per run</h2>
<div class="muted">Run 1 is the first call at the clip's length (cold), the others are warm.</div>
<div class="controls">
  <fieldset><legend>Backend</legend><span id="backends"></span></fieldset>
  <fieldset><legend>Run</legend><span id="runs"></span></fieldset>
  <fieldset><legend>Device</legend><span id="devices"></span></fieldset>
  <fieldset><legend>Clip</legend><span id="clips"></span></fieldset>
  <fieldset><legend>Scale</legend><label><input type="checkbox" id="log" checked> log</label></fieldset>
</div>
<svg id="runtime" width="1000" height="420"></svg>

<h2>Lines of code per backend</h2>
<div class="muted">Non-blank lines in each backend's package under tinybark/ (onnx: tinybark/ort/), without tinygrad's attempts/. pytorch-compile shares pytorch's code; the shared audio, text and weights code isn't counted.</div>
<svg id="loc" width="1000" height="320"></svg>
<div id="tip"></div>

<script>
const DATA = /*DATA*/null;
const NS = "http://www.w3.org/2000/svg";
const RUN_COLORS = ["#d1495b", "#2e86ab", "#66a182", "#edae49", "#8d6a9f", "#555"];
const runCount = Math.max(...DATA.results.flatMap(r => Object.values(r.clips).map(c => c.length)));
const state = {
  backends: new Set(DATA.backends),
  runs: new Set([...Array(runCount).keys()]),
  device: "gpu",
  clip: Object.keys(DATA.clips)[0],
  log: true,
};

document.getElementById("machine").textContent =
  `${DATA.machine.cpu} · kernel ${DATA.machine.kernel} · latest result per backend and device`;

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function control(id, type, items, checked, onChange, swatch) {
  const host = document.getElementById(id);
  items.forEach(([value, text], i) => {
    const label = document.createElement("label");
    const input = Object.assign(document.createElement("input"), { type, name: id, checked: checked(value) });
    input.addEventListener("change", () => { onChange(value, input.checked); draw(); });
    label.append(input, " ");
    if (swatch) label.append(Object.assign(document.createElement("span"), { className: "swatch", style: `background:${swatch[i]}` }));
    label.append(text);
    host.append(label);
  });
}

const toggle = set => (value, on) => on ? set.add(value) : set.delete(value);
control("backends", "checkbox", DATA.backends.map(b => [b, b]), b => state.backends.has(b), toggle(state.backends));
control("runs", "checkbox", [...Array(runCount).keys()].map(i => [i, `${i + 1}${i === 0 ? " (cold)" : ""}`]), i => state.runs.has(i), toggle(state.runs), RUN_COLORS);
control("devices", "radio", [["gpu", "GPU"], ["cpu", "CPU"]], d => d === state.device, d => state.device = d);
control("clips", "radio", Object.entries(DATA.clips).map(([name, c]) => [name, `${name} (${c.seconds.toFixed(2)} s)`]), c => c === state.clip, c => state.clip = c);
document.getElementById("log").addEventListener("change", e => { state.log = e.target.checked; draw(); });

const tip = document.getElementById("tip");
function hover(node, html) {
  node.addEventListener("mousemove", e => { tip.innerHTML = html; tip.style.display = "block"; tip.style.left = e.clientX + 12 + "px"; tip.style.top = e.clientY + 12 + "px"; });
  node.addEventListener("mouseleave", () => tip.style.display = "none");
}

function fmt(s) { return s >= 10 ? s.toFixed(1) : s >= 1 ? s.toFixed(2) : s.toFixed(3); }

// groups: [{name, bars: [{value, color, tip}]}]
function barChart(svg, groups, { log, unit, yLabel }) {
  svg.replaceChildren();
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height");
  const m = { top: 36, right: 20, bottom: 40, left: 64 };
  const w = W - m.left - m.right, h = H - m.top - m.bottom;
  const g = el("g", { transform: `translate(${m.left},${m.top})` }, svg);
  const values = groups.flatMap(gr => gr.bars.map(b => b.value)).filter(v => v > 0);
  if (!values.length) {
    el("text", { x: w / 2, y: h / 2, "text-anchor": "middle", fill: "#6e6e73" }, g).textContent = "Nothing selected";
    return;
  }
  let y, ticks;
  if (log) {
    const lo = Math.floor(Math.log10(Math.min(...values))), hi = Math.ceil(Math.log10(Math.max(...values)));
    const top = hi === lo ? hi + 1 : hi;
    y = v => h - (Math.log10(v) - lo) / (top - lo) * h;
    ticks = []; for (let p = lo; p <= top; p++) ticks.push(10 ** p);
  } else {
    const max = Math.max(...values), step = 10 ** Math.floor(Math.log10(max));
    const nice = [1, 2, 5, 10].map(k => k * step).find(s => max / s <= 6) ?? step;
    const top = Math.ceil(max / nice) * nice;
    y = v => h - v / top * h;
    ticks = []; for (let t = 0; t <= top + 1e-9; t += nice) ticks.push(t);
  }
  const base = log ? h : y(0);
  for (const t of ticks) {
    el("line", { x1: 0, x2: w, y1: y(t), y2: y(t), stroke: "#e4e4e8" }, g);
    el("text", { x: -8, y: y(t) + 4, "text-anchor": "end" }, g).textContent = unit === "s" ? +t.toPrecision(3) + " s" : t.toLocaleString();
  }
  el("line", { x1: 0, x2: 0, y1: 0, y2: h, stroke: "#1d1d1f", "stroke-width": 2 }, g);
  el("line", { x1: 0, x2: w, y1: h, y2: h, stroke: "#1d1d1f", "stroke-width": 2 }, g);
  const label = el("text", { x: -m.left + 4, y: -20, fill: "#6e6e73" }, g);
  label.textContent = yLabel;

  const slot = w / groups.length;
  groups.forEach((gr, i) => {
    const cx = slot * (i + 0.5);
    const bw = Math.min(36, (slot * 0.7) / Math.max(1, gr.bars.length));
    const x0 = cx - (bw * gr.bars.length) / 2;
    gr.bars.forEach((b, j) => {
      const top = b.value > 0 ? y(b.value) : base;
      const r = el("rect", { class: "bar", x: x0 + j * bw + 1, y: top, width: bw - 2, height: Math.max(0, base - top), fill: b.color, rx: 2 }, g);
      hover(r, b.tip);
      if (b.label) el("text", { x: x0 + j * bw + bw / 2, y: top - 5, "text-anchor": "middle", "font-size": 11 }, g).textContent = b.label;
    });
    el("text", { x: cx, y: h + 20, "text-anchor": "middle" }, g).textContent = gr.name;
    if (gr.note) el("text", { x: cx, y: h + 34, "text-anchor": "middle", fill: "#6e6e73", "font-size": 11 }, g).textContent = gr.note;
  });
}

function draw() {
  const runs = [...state.runs].sort();
  const groups = DATA.backends.filter(b => state.backends.has(b)).map(name => {
    const r = DATA.results.find(r => r.backend === name && r.device === state.device);
    const times = r?.clips[state.clip] ?? [];
    return {
      name,
      note: r ? (r.match ? "" : "tokens differ") : "no result",
      bars: runs.filter(i => i < times.length).map(i => ({
        value: times[i],
        color: RUN_COLORS[i % RUN_COLORS.length],
        label: runs.length <= 3 ? fmt(times[i]) : "",
        tip: `<b>${name}</b>, run ${i + 1}${i === 0 ? " (cold)" : ""}: ${fmt(times[i])} s<br>${r.description}<br>load ${r.load != null ? fmt(r.load) + " s" : "-"} · ${r.file}`,
      })),
    };
  });
  barChart(document.getElementById("runtime"), groups, { log: state.log, unit: "s", yLabel: "Runtime" });
}

barChart(document.getElementById("loc"), DATA.loc.map(b => ({
  name: b.backend,
  bars: [{
    value: b.lines,
    color: "#2e86ab",
    label: String(b.lines),
    tip: `<b>${b.backend}</b>: ${b.lines} lines<br>` + Object.entries(b.files).map(([f, n]) => `${f}: ${n}`).join("<br>"),
  }],
})), { log: false, unit: "lines", yLabel: "Lines of code" });

draw();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
