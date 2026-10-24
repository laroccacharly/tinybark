"""Log-mel features, a port of tinybark's audio.py (transformers' ParakeetFeatureExtractor).
The 512-point real FFT is a matmul with a DFT basis."""

from std.math import cos, exp, log, pi, sin, sqrt

from .tensor import Tensor, linear

comptime SAMPLE_RATE = 16_000
comptime N_FFT = 512
comptime HOP = 160
comptime WIN = 400
comptime N_MELS = 128
comptime BINS = N_FFT // 2 + 1
comptime PREEMPHASIS: Float32 = 0.97
comptime LOG_GUARD: Float32 = 1.0 / Float32(1 << 24)
comptime EPSILON: Float32 = 1e-5


def _hz_to_mel(f: Float64) -> Float64:
    if f >= 1000.0:
        return 15.0 + log(max(f, 1e-10) / 1000.0) / (log(6.4) / 27.0)
    return 3.0 * f / 200.0


def _mel_to_hz(m: Float64) -> Float64:
    if m >= 15.0:
        return 1000.0 * exp(log(6.4) / 27.0 * (m - 15.0))
    return 200.0 * m / 3.0


def mel_filters() -> Tensor:
    """librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm="slaney"), shape (N_MELS, BINS)."""
    var top = _hz_to_mel(Float64(SAMPLE_RATE) / 2)
    var mels = List[Float64]()
    for i in range(N_MELS + 2):
        mels.append(_mel_to_hz(top * Float64(i) / Float64(N_MELS + 1)))
    var out = Tensor(N_MELS, BINS)
    for m in range(N_MELS):
        var scale = 2.0 / (mels[m + 2] - mels[m])
        for b in range(BINS):
            var f = Float64(SAMPLE_RATE) / 2 * Float64(b) / Float64(BINS - 1)
            var lower = (f - mels[m]) / (mels[m + 1] - mels[m])
            var upper = (mels[m + 2] - f) / (mels[m + 2] - mels[m + 1])
            out[m, b] = Float32(max(0.0, min(lower, upper)) * scale)
    return out^


def dft_basis() -> Tensor:
    """(2 * BINS, N_FFT): cos rows then sin rows, with torch's symmetric
    400-sample Hann window centred in the frame folded in."""
    var out = Tensor.zeros(2 * BINS, N_FFT)
    var offset = (N_FFT - WIN) // 2
    for n in range(WIN):
        var window = 0.5 - 0.5 * cos(2.0 * pi * Float64(n) / Float64(WIN - 1))
        var t = n + offset
        for k in range(BINS):
            var angle = 2.0 * pi * Float64(k * t % N_FFT) / Float64(N_FFT)
            out[k, t] = Float32(window * cos(angle))
            out[BINS + k, t] = Float32(-window * sin(angle))
    return out^


struct Features(Movable):
    var mel: Tensor
    var basis: Tensor

    def __init__(out self):
        self.mel = mel_filters()
        self.basis = dft_basis()

    def __call__(self, samples: List[Float32]) raises -> Tensor:
        """Normalized log-mel features, shape (frames, N_MELS), for 16 kHz mono samples."""
        var count = len(samples)
        var frames = count // HOP
        var padded = List[Float32](length=count + N_FFT, fill=0)
        padded[N_FFT // 2] = samples[0]
        for i in range(1, count):
            padded[N_FFT // 2 + i] = samples[i] - PREEMPHASIS * samples[i - 1]
        var windows = Tensor(frames, N_FFT)
        for f in range(frames):
            for j in range(N_FFT):
                windows[f, j] = padded[f * HOP + j]
        var spectrum = linear(windows, self.basis)
        var power = Tensor(frames, BINS)
        for f in range(frames):
            for k in range(BINS):
                var re = spectrum[f, k]
                var im = spectrum[f, BINS + k]
                power[f, k] = re * re + im * im
        var mel = linear(power, self.mel)
        for i in range(mel.size()):
            mel.ptr[unsafe_offset=i] = log(mel.ptr[unsafe_offset=i] + LOG_GUARD)
        for m in range(N_MELS):
            var total: Float32 = 0
            for f in range(frames):
                total += mel[f, m]
            var mean = total / Float32(frames)
            var spread: Float32 = 0
            for f in range(frames):
                spread += (mel[f, m] - mean) ** 2
            var std = sqrt(spread / Float32(frames - 1))
            for f in range(frames):
                mel[f, m] = (mel[f, m] - mean) / (std + EPSILON)
        return mel^


def read_pcm(path: String) raises -> List[Float32]:
    """16 kHz s16le mono PCM as float32 samples in [-1, 1)."""
    var data = open(path, "r").read_bytes()
    var out = List[Float32](capacity=len(data) // 2)
    for i in range(len(data) // 2):
        var v = Int16(Int(data[2 * i]) | (Int(data[2 * i + 1]) << 8))
        out.append(Float32(v) / 32768)
    return out^
