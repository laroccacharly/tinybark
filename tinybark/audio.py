"""Log-mel features, a NumPy port of transformers' ParakeetFeatureExtractor."""

import numpy as np

SAMPLE_RATE = 16_000
N_FFT, HOP, WIN = 512, 160, 400
N_MELS = 128
PREEMPHASIS = 0.97
LOG_GUARD = 2**-24
EPSILON = 1e-5


def _mel_filters() -> np.ndarray:
    """librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm="slaney"), shape (n_mels, n_fft // 2 + 1)."""

    def hz_to_mel(f):
        f = np.asarray(f, np.float64)
        log = f >= 1000.0
        return np.where(log, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), 3.0 * f / 200.0)

    def mel_to_hz(m):
        log = m >= 15.0
        return np.where(log, 1000.0 * np.exp(np.log(6.4) / 27.0 * (m - 15.0)), 200.0 * m / 3.0)

    fft_freqs = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    mel_freqs = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2))
    ramps = mel_freqs[:, None] - fft_freqs[None, :]
    widths = np.diff(mel_freqs)
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_freqs[2:] - mel_freqs[:-2]))[:, None]
    return weights.astype(np.float32)


MEL_FILTERS = _mel_filters()
# torch.hann_window(400, periodic=False), centred in the 512-sample frame as torch.stft does.
WINDOW = np.zeros(N_FFT, np.float32)
WINDOW[(N_FFT - WIN) // 2 : (N_FFT + WIN) // 2] = np.hanning(WIN)


def features(samples: np.ndarray) -> np.ndarray:
    """Normalized log-mel features, shape (frames, N_MELS), for 16 kHz mono float32 samples."""
    x = samples.astype(np.float32)
    x = np.concatenate([x[:1], x[1:] - PREEMPHASIS * x[:-1]])
    x = np.pad(x, N_FFT // 2)
    frames = len(samples) // HOP
    # The last centred frame is padding: the extractor masks it out.
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    spectrum = np.fft.rfft(x[idx] * WINDOW, axis=-1)
    power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
    mel = np.log(power @ MEL_FILTERS.T + LOG_GUARD)
    mean = mel.mean(0)
    std = np.sqrt(((mel - mean) ** 2).sum(0) / (frames - 1))
    return ((mel - mean) / (std + EPSILON)).astype(np.float32)
