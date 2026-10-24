"""The pieces the benchmark relies on, without a GPU. Backends' tokens are checked
against the references by the benchmark itself (python -m bench)."""

from bench.__main__ import report
from bench.worker import NAMES, clip
from tinybark import weights
from tinybark.backends import BACKENDS, DEVICES
from tinybark.text import Tokenizer


def test_references_decode_to_their_text():
    tokenizer = Tokenizer(weights.tokenizer_file())
    for name in NAMES:
        samples, reference = clip(name)
        assert tokenizer.decode(reference["tokens"]) == reference["text"]
    assert len(clip("quilter3")[0]) == 3 * len(clip("quilter")[0])


def test_report_flags_mismatches_and_errors():
    run = lambda seconds, match: {"seconds": seconds, "match": match, "tokens": []}
    results = [
        {"backend": "a", "device": "gpu", "load_seconds": 1.0, "error": None, "clips": {"quilter": [run(5.0, True), run(0.1, True), run(0.3, True)]}},
        {"backend": "b", "device": "cpu", "load_seconds": 2.0, "error": "exited with -6", "clips": {"quilter": [run(0.2, False)]}},
        {"backend": "c", "device": "gpu", "load_seconds": None, "error": "timed out after 1800 s", "clips": {}},
    ]
    table = report(results).splitlines()
    assert table[2] == "| a | gpu | 1.00 | quilter | 5.000 | 0.200 | 0.100 | match |"
    assert table[3] == "| b | cpu | 2.00 | quilter | 0.200 | - | - | differ |"
    assert "exited with -6" in table[4] and "timed out" in table[5]


def test_every_backend_runs_on_gpu_and_cpu():
    assert set(BACKENDS) == {"pytorch", "pytorch-compile", "tinygrad", "onnx", "mojo"}
    assert all(set(b.descriptions) == set(DEVICES) for b in BACKENDS.values())
    assert BACKENDS["tinygrad"].env("gpu") == {"DEV": "AMD"} and BACKENDS["tinygrad"].env("cpu") == {"DEV": "CPU"}
    assert BACKENDS["onnx"].env("cpu") == {}
