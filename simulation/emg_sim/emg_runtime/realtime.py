"""Real-time decoder: the exact causal pipeline intended for the device.

  EMG block (1024 Hz, 24 ch) -> streaming HP + notch filter -> ring buffer
  -> divide by frozen train RMS -> model (ONNX Runtime or torch) -> softmax
  -> confidence gate (below threshold => REST) -> causal smoothing -> command

Every step uses only past samples, so the decision at time t depends on data <= t.
"""
import collections
import time

import numpy as np

from . import signal

# class -> prosthetic command (8 putEMG classes)
COMMANDS = {
    "rest": {"action": "HOLD", "note": "keep current posture / relax motors"},
    "fist": {"action": "POWER_GRASP", "fingers": "all flex", "thumb": "flex + oppose"},
    "flexion": {"action": "WRIST_FLEX or CLOSE_HAND", "note": "map to close if no wrist actuator"},
    "extension": {"action": "WRIST_EXT or OPEN_HAND", "note": "map to open if no wrist actuator"},
    "pinch_index": {"action": "PRECISION_PINCH", "fingers": "thumb + index"},
    "pinch_middle": {"action": "TRIPOD", "fingers": "thumb + index + middle"},
    "pinch_ring": {"action": "INDEX_POINT / KEY_GRIP", "note": "user-assignable"},
    "pinch_little": {"action": "THUMB_ONLY / MODE_SWITCH", "note": "user-assignable"},
}


class RealtimeDecoder:
    def __init__(self, infer, fs, n_ch, window_ms, step_ms, scale, preprocess, class_names,
                 conf_threshold=0.6, smooth="majority", smooth_len=5, ema_alpha=0.3):
        self.infer = infer                       # callable: (1, L, C) float32 -> (1, K) probabilities
        self.L = int(round(window_ms * fs / 1000))
        self.step = int(round(step_ms * fs / 1000))
        self.filt = signal.StreamingFilter(signal.design_chain(fs, **preprocess), n_ch)
        self.buf = np.zeros((self.L, n_ch), np.float32)
        self.filled = 0
        self.since = 0
        self.scale = np.asarray(scale, np.float32)
        self.names = class_names
        self.thr, self.smooth, self.k, self.alpha = conf_threshold, smooth, smooth_len, ema_alpha
        self.hist = collections.deque(maxlen=smooth_len)
        self.ema = None
        self.compute_ms = []

    def push(self, block):
        """Feed a block of raw samples (n, C). Returns a list with 0 or 1 decision dicts."""
        out = []
        y = self.filt(np.asarray(block, np.float64))
        n = len(y)                       # ring buffer update
        if n >= self.L:
            self.buf[:] = y[-self.L:]
        else:
            self.buf[:-n] = self.buf[n:]
            self.buf[-n:] = y
        self.filled += n
        self.since += n
        if self.filled >= self.L and self.since >= self.step:
            self.since = 0
            out.append(self.decide())
        return out

    def decide(self):
        t0 = time.perf_counter()
        p = self.infer((self.buf / self.scale)[None])[0]
        raw = int(p.argmax()) if p.max() >= self.thr else 0
        if self.smooth == "ema":
            self.ema = p if self.ema is None else self.alpha * p + (1 - self.alpha) * self.ema
            dec = int(self.ema.argmax()) if self.ema.max() >= self.thr else 0
        elif self.smooth == "debounce":
            cur = getattr(self, "cur", 0)
            if raw == cur:
                self.cnt = 0
            elif raw == getattr(self, "cand", None):
                self.cnt += 1
            else:
                self.cand, self.cnt = raw, 1
            if self.cnt >= self.k:
                self.cur, self.cnt = self.cand, 0
            dec = getattr(self, "cur", 0)
        elif self.smooth == "majority":
            self.hist.append(raw)
            # tie-break identical to metrics.decide (lowest class id)
            c = collections.Counter(self.hist)
            top = max(c.values())
            dec = min(k for k, v in c.items() if v == top)
        else:
            dec = raw
        self.compute_ms.append((time.perf_counter() - t0) * 1000)
        name = self.names[dec]
        return {"class": dec, "name": name, "conf": float(p.max()), "probs": p, "command": COMMANDS[name]}
