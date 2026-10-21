"""Causal signal processing. Every filter here is a forward-only IIR (sosfilt) so
the offline pipeline is sample-for-sample identical to what runs on the device."""
import numpy as np
from scipy import signal as ss


def aa_decimate(x, fs, decim, cutoff):
    """Causal elliptic anti-alias low-pass then integer decimation. x: (N, C)."""
    sos = ss.ellip(8, 0.1, 60, cutoff, fs=fs, output="sos")
    y = ss.sosfilt(sos, x, axis=0)
    return y[::decim]


def design_chain(fs, highpass=20, hp_order=4, lowpass=None, notch=(), notch_q=30, **_):
    """List of SOS sections applied in order (HP, optional LP, mains notches)."""
    secs = []
    if highpass:
        secs.append(ss.butter(hp_order, highpass, "highpass", fs=fs, output="sos"))
    if lowpass:
        secs.append(ss.butter(4, lowpass, "lowpass", fs=fs, output="sos"))
    for f0 in notch or ():
        if f0 < fs / 2:
            b, a = ss.iirnotch(f0, notch_q, fs=fs)
            secs.append(ss.tf2sos(b, a))
    return np.vstack(secs) if secs else None


def apply_chain(x, sos):
    """Causal filtering of a whole recording (N, C) -> float32."""
    if sos is None:
        return (x - x[:1]).astype(np.float32)
    return ss.sosfilt(sos, x.astype(np.float64), axis=0).astype(np.float32)


class StreamingFilter:
    """Same chain, sample-block streaming with persistent state (for the device loop)."""

    def __init__(self, sos, n_ch):
        self.sos = sos
        self.zi = np.zeros((sos.shape[0], 2, n_ch)) if sos is not None else None

    def __call__(self, block):
        if self.sos is None:
            return block.astype(np.float32)
        y, self.zi = ss.sosfilt(self.sos, block, axis=0, zi=self.zi)
        return y.astype(np.float32)
