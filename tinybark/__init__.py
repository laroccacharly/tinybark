"""Underdog Bark 3.0 Clean speech-to-text on four backends, one folder each:
tinybark.tinygrad, tinybark.ort (ONNX Runtime), tinybark.pytorch and tinybark.mojo.
tinybark.backends loads any of them by name.

    bark = tinybark.load()  # tinygrad, warmed up; see tinybark.tinygrad
    text = bark.transcribe(samples)  # 16 kHz mono float32 NumPy array
"""

__all__ = ["Bark", "load"]


def __getattr__(name: str):
    # Lazy, so importing one backend doesn't import tinygrad.
    if name in __all__:
        from . import tinygrad

        return getattr(tinygrad, name)
    raise AttributeError(name)
