#!/usr/bin/env python3
"""Block diagram and flowchart for the report (matplotlib only)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
C = {"sense": "#dbeafe", "proc": "#dcfce7", "ml": "#fde68a", "mcu": "#fed7aa", "act": "#e9d5ff", "sim": "#e5e7eb"}


def box(ax, x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#333", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x0, y0, x1, y1, label=None):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.3))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label, ha="center", fontsize=7, color="#444")


def block_diagram():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    w, h, y = 1.85, 1.0, 3.6
    xs = [0.2, 2.35, 4.5, 6.65, 8.8, 10.95]
    labels = [("Forearm sEMG\n24 electrodes\n(putEMG, 1024 Hz)", "sense"),
              ("Signal conditioning\n20 Hz HP (4th order)\n50 Hz + harmonic notches", "proc"),
              ("Windowing\n300 ms window\n50 ms step, RMS scale", "proc"),
              ("CNN-LSTM\n(ONNX, 97.5 k params)\n8-class softmax", "ml"),
              ("Command filter\ndebounce 200 ms /\nconf ≥ 0.7 + majority", "ml"),
              ("Gesture → 16\nservo angles\n(lookup table)", "mcu")]
    for x, (t, c) in zip(xs, labels):
        box(ax, x, y, w, h, t, C[c])
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + w, y + h / 2, b, y + h / 2)
    # lower row
    box(ax, 10.95, 1.6, w, h, "Microcontroller\n(Arduino / ESP32)\nrate limit + watchdog", C["mcu"])
    box(ax, 8.8, 1.6, w, h, "PCA9685\n16-ch 50 Hz PWM\n1000–2000 µs", C["mcu"])
    box(ax, 6.65, 1.6, w, h, "16 hobby servos\ntendon + pulley\nBiDexHand", C["act"])
    box(ax, 6.65, 0.1, w, 1.0, "MuJoCo simulation\nBiDexHand URDF\n(same 16 angles)", C["sim"])
    arrow(ax, xs[5] + w / 2, y, 10.95 + w / 2, 1.6 + h, "UART 'G<id>'")
    arrow(ax, 10.95, 2.1, 8.8 + w, 2.1, "I²C")
    arrow(ax, 8.8, 2.1, 6.65 + w, 2.1, "PWM")
    ax.plot([xs[5] + w, 12.95, 12.95, 8.8 + 0.4], [y + 0.3, y + 0.3, 0.6, 0.6], color="#333", lw=1.3)
    arrow(ax, 8.8 + 0.4, 0.6, 6.65 + w, 0.6)
    ax.text(10.3, 0.75, "simulation path (this project)", ha="center", fontsize=8, color="#555")
    ax.text(0.2, 2.6, "Host processor (laptop / Raspberry Pi)", fontsize=9, color="#555")
    ax.add_patch(plt.Rectangle((0.1, 3.35), 10.75, 1.5, fill=False, ls="--", ec="#999"))
    ax.text(0.15, 4.95, "Real-time EMG decoding pipeline (causal, runs every 50 ms)", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig_block_diagram.png", dpi=150)
    plt.close(fig)


def flowchart():
    steps = [("START", "#e5e7eb"), ("Initialise: load ONNX model, scales,\nfilter states, BiDexHand model", C["proc"]),
             ("Read next 50 ms EMG block\n(51 samples × 24 ch)", C["sense"]),
             ("Causal HP + notch filter\n(state carried between blocks)", C["proc"]),
             ("Ring buffer full (300 ms)?", "#fff"),
             ("Normalise ÷ train RMS → CNN-LSTM\n→ softmax probabilities", C["ml"]),
             ("Command filter: same class for\n4 consecutive decisions?", "#fff"),
             ("Look up 16 servo angles for gesture;\nsend 'G<id>' to MCU", C["mcu"]),
             ("MCU: move servos at ≤ 600 °/s;\nno command for 2 s → REST", C["mcu"]),
             ("Update simulation + GUI", C["sim"])]
    fig, ax = plt.subplots(figsize=(7, 12))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 12)
    ax.axis("off")
    w, h = 4.4, 0.8
    x = 1.3
    ys = [11.0 - i * 1.15 for i in range(len(steps))]
    for (t, c), y in zip(steps, ys):
        if c == "#fff":
            ax.add_patch(plt.Polygon([[x + w / 2, y + h / 2 + 0.25], [x + w + 0.3, y], [x + w / 2, y - h / 2 - 0.25 + 0.4 - 0.4],
                                      [x - 0.3, y]], closed=True, fc="#fef9c3", ec="#333"))
            ax.text(x + w / 2, y, t, ha="center", va="center", fontsize=8.5)
        else:
            box(ax, x, y - h / 2, w, h, t, c, fs=8.5)
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, x + w / 2, a - h / 2 - (0.05 if a != ys[4] and a != ys[6] else 0.15), x + w / 2, b + h / 2 + 0.02)
    ax.text(x + w / 2 + 0.15, ys[4] - 0.75, "yes", fontsize=8)
    ax.text(x + w / 2 + 0.15, ys[6] - 0.75, "yes", fontsize=8)
    # loops back
    for yy, lab in [(ys[4], "no"), (ys[6], "no: hold previous")]:
        ax.plot([x + w + 0.3, 6.6, 6.6], [yy, yy, ys[2]], color="#333", lw=1.1)
        ax.annotate("", (x + w, ys[2]), (6.6, ys[2]), arrowprops=dict(arrowstyle="-|>", color="#333"))
        ax.text(6.55, yy + 0.12, lab, fontsize=7.5, ha="right", bbox=dict(fc="white", ec="none", pad=1))
    ax.plot([x, 0.5, 0.5], [ys[-1], ys[-1], ys[2]], color="#333", lw=1.1)
    ax.annotate("", (x, ys[2]), (0.5, ys[2]), arrowprops=dict(arrowstyle="-|>", color="#333"))
    ax.text(0.55, (ys[-1] + ys[2]) / 2, "every 50 ms", rotation=90, fontsize=8, va="center")
    fig.tight_layout()
    fig.savefig(FIG / "fig_flowchart.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    block_diagram()
    flowchart()
