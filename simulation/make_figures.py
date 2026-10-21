#!/usr/bin/env python3
"""Regenerate the results and figures used in docs/report.

  python simulation/make_figures.py [--data DIR]

Runs the exact streaming decoder of the app (stable filter) over every recording in
DIR (default simulation/data), writes docs/report/results.json and the figures.
"""
import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app import MLEngine  # noqa: E402
from emg_sim.bidexhand import ALL_POSES, SERVO_NAMES, BiDexHand  # noqa: E402

OUT = HERE.parent / "docs" / "report"
FIG = OUT / "figures"


def evaluate(paths):
    ml = MLEngine(HERE / "model")
    K = len(ml.names)
    cm = np.zeros((K, K), int)
    per_rec, timeline = [], None
    lat = []
    for p in paths:
        ml.load(p)
        dec = ml.dec
        dec.thr, dec.smooth, dec.k = 0.0, "debounce", 4       # "stable" filter of the app
        dec.cur, dec.cand, dec.cnt = 0, None, 0
        dec.hist = collections.deque(maxlen=4)
        pred, truth = [], []
        while True:
            d = ml.tick()
            if d is None:
                break
            if d.get("warming"):
                continue
            pi, ti = ml.names.index(d["name"]), ml.names.index(d["truth"])
            cm[ti, pi] += 1
            pred.append(pi)
            truth.append(ti)
        lat += dec.compute_ms
        pred, truth = np.array(pred), np.array(truth)
        per_rec.append({"recording": Path(p).name, "subject": ml.subject, "session": ml.session,
                        "decisions": int(len(pred)), "accuracy": float((pred == truth).mean())})
        if timeline is None:
            timeline = (pred, truth, ml.names)
        print(per_rec[-1])
    rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    res = {"filter": "stable (debounce 4 x 50 ms)", "recordings": per_rec,
           "decisions": int(cm.sum()), "accuracy": float(cm.trace() / cm.sum()),
           "balanced_accuracy": float(rec.mean()), "macro_f1": float(f1.mean()),
           "per_class": {n: {"recall": float(r), "precision": float(q), "f1": float(f)}
                         for n, r, q, f in zip(ml.names, rec, prec, f1)},
           "inference_ms_mean": float(np.mean(lat)), "inference_ms_p99": float(np.percentile(lat, 99)),
           "confusion_matrix": cm.tolist(), "class_names": ml.names}
    return res, timeline


def fig_confusion(res):
    cm = np.array(res["confusion_matrix"], float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = res["class_names"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(n)):
        for j in range(len(n)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(n)), n, rotation=45, ha="right")
    ax.set_yticks(range(len(n)), n)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"Streaming decisions on held-out subjects (row-normalised)\n"
                 f"accuracy {res['accuracy']:.2f} · balanced {res['balanced_accuracy']:.2f} · macro-F1 {res['macro_f1']:.2f}")
    fig.tight_layout()
    fig.savefig(FIG / "fig_confusion_stream.png", dpi=150)
    plt.close(fig)


def fig_timeline(tl):
    pred, truth, names = tl
    t = np.arange(len(pred)) * 0.05
    sl = slice(0, min(len(t), 2400))       # first 120 s
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.step(t[sl], truth[sl], where="post", color="#1baf7a", lw=2.5, label="ground truth")
    ax.step(t[sl], pred[sl], where="post", color="#2a78d6", lw=1, label="CNN-LSTM decision (stable filter)")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Gesture decisions over time — unseen subject S23, session 1")
    fig.tight_layout()
    fig.savefig(FIG / "fig_timeline.png", dpi=150)
    plt.close(fig)


def fig_servo_response():
    h = BiDexHand()
    seq = [("rest", 0.3), ("fist", 0.6), ("rest", 0.6), ("pinch_index", 0.6), ("rest", 0.4)]
    t, tgt, act = [], [], []
    for g, dur in seq:
        h.command(ALL_POSES[g]["servo"])
        for _ in range(int(dur / 0.005)):
            h.step(0.005)
            t.append(h.time)
            tgt.append(h.target.copy())
            act.append(h.servo.copy())
    t, tgt, act = np.array(t), np.array(tgt), np.array(act)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for ch, c in [(6, "#2a78d6"), (11, "#eb6834"), (15, "#1baf7a")]:
        ax.plot(t, tgt[:, ch], "--", color=c, lw=1)
        ax.plot(t, act[:, ch], color=c, lw=2, label=SERVO_NAMES[ch])
    x = 0.3
    for g, dur in seq[1:]:
        ax.axvline(x, color="#bbb", lw=0.8)
        ax.text(x + 0.01, 92, g, fontsize=8, color="#555")
        x += dur
    ax.set_ylim(-75, 100)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("servo angle (deg)")
    ax.set_title("Simulated servo response (dashed = command, solid = servo, 600 deg/s limit, tau 40 ms)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_servo_response.png", dpi=150)
    plt.close(fig)


def fig_gestures():
    h = BiDexHand()
    r = mujoco.Renderer(h.m, 300, 300)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    tiles = []
    for n, g in ALL_POSES.items():
        h.command(g["servo"])
        h.step(1.0)
        c = mujoco.MjvCamera()
        c.lookat[:] = [-0.02, -0.01, 0.09]
        c.distance, c.azimuth, c.elevation = 0.32, 30, -15
        r.update_scene(h.d, c)
        im = Image.fromarray(r.render())
        ImageDraw.Draw(im).text((8, 8), n, fill=(255, 220, 0), font=font)
        tiles.append(np.array(im))
    Image.fromarray(np.vstack([np.hstack(tiles[i:i + 5]) for i in (0, 5)])).save(FIG / "fig_gestures.png")
    r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data"))
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    res, tl = evaluate(paths)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    fig_confusion(res)
    fig_timeline(tl)
    fig_servo_response()
    fig_gestures()
    print(json.dumps({k: res[k] for k in ("decisions", "accuracy", "balanced_accuracy", "macro_f1", "inference_ms_mean")}, indent=1))


if __name__ == "__main__":
    main()
