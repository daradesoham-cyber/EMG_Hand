#!/usr/bin/env python3
"""EMG-controlled BiDexHand — microcontroller project simulation.

One window:
  * 3D MuJoCo simulation of the 16-servo BiDexHand (same URDF/meshes as the ROS 2 RViz demo)
  * ML MODE    : recorded putEMG sEMG (24 ch, 1024 Hz) from a subject NEVER used in training
                 -> causal filter -> CNN-LSTM (ONNX) -> probabilities -> command filter
                 -> gesture -> 16 servo angles / PWM pulse widths -> fingers move.
                 Ground truth is shown next to every prediction (errors stay visible).
  * MANUAL MODE: presenter picks the gesture; no ML involved (clearly labelled).
  * Optional --serial PORT: every decided gesture is also sent to the microcontroller
                 firmware (firmware/emg_hand_servo) as "G<id>\\n", so the real servo board
                 follows the simulation.

Run:  python simulation/app.py            (see simulation/README.md)
"""
import collections
import glob
import json
import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageTk  # noqa: E402

from emg_sim.bidexhand import ALL_POSES, GESTURES, SERVO_NAMES, BiDexHand  # noqa: E402

BLUE, ORANGE, GREEN, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#898781"
BG, PANEL, INK, INK2 = "#f4f4f1", "#fcfcfb", "#0b0b0b", "#52514e"
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except OSError:
    FONT = FONT_S = ImageFont.load_default()
STEP_MS = 50                                     # decision period (20 Hz)
GESTURE_IDS = {g: i for i, g in enumerate(ALL_POSES)}   # serial protocol ids (same order as firmware table)


class MLEngine:
    """Loads the exported ONNX classifier + deploy.json and replays a recording."""

    def __init__(self, model_dir):
        import onnxruntime as ort
        from emg_sim.emg_runtime.realtime import RealtimeDecoder
        self.dep = json.load(open(model_dir / "deploy.json"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        onnx = model_dir / f"{self.dep['model']}.onnx"
        self.sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
        self.names = self.dep["class_names"]
        self.Decoder = RealtimeDecoder
        self.rec = None

    def infer(self, x):
        lo = self.sess.run(None, {"emg": x.astype(np.float32)})[0]
        e = np.exp(lo - lo.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def load(self, path):
        z = np.load(path)
        self.emg = z["emg"].astype(np.float32)
        self.gt = z["TRAJ_GT"].astype(int)
        self.subject, self.session = int(z["subject"]), int(z["session"])
        pp = self.dep["postproc"]
        self.dec = self.Decoder(self.infer, self.dep["fs"], 24, self.dep["window_ms"], self.dep["step_ms"],
                                self.dep["scale"], self.dep["preprocess"], self.names,
                                pp.get("conf_threshold", 0.0), pp.get("smooth", "none"),
                                int(pp.get("smooth_len", 1)), pp.get("ema_alpha", 0.3))
        self.pos = 0
        self.n_ok = self.n_tot = 0
        self.rec = path

    def gt_name(self, i):
        code = int(self.gt[min(i, len(self.gt) - 1)])
        lut = {-1: 0, 0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 7: 5, 8: 6, 9: 7}
        return self.names[lut.get(code, 0)]

    def next_onset(self, after):
        """Sample index where the next gesture segment starts in the ground truth.
        Used ONLY for navigation (skipping rest); predictions never see the labels."""
        g = self.gt[after:] > 0
        on = np.flatnonzero(g[1:] & ~g[:-1])
        return after + int(on[0]) + 1 if len(on) else None

    def fast_forward(self, target):
        """Feed all EMG up to `target` through the decoder (nothing is skipped for the model)."""
        last = None
        while self.pos < target:
            d = self.tick()
            if d is None:
                break
            if not d.get("warming"):
                last = d
        return last

    def tick(self):
        """Advance one decision period of recorded EMG. Returns a decision dict or None at the end."""
        n = self.dec.step
        if self.pos + n > len(self.emg):
            return None
        out = self.dec.push(self.emg[self.pos:self.pos + n])
        self.pos += n
        if not out:
            return {"warming": True}
        d = out[0]
        d["raw"] = self.names[int(np.argmax(d["probs"]))]
        d["truth"] = self.gt_name(self.pos - 1)
        self.n_tot += 1
        self.n_ok += d["name"] == d["truth"]
        return d


class SerialLink:
    """Sends the decided gesture to the microcontroller firmware: 'G<id>\\n' on change,
    and again every 0.5 s as a heartbeat for the firmware's 2 s watchdog."""

    def __init__(self, port, baud=115200):
        import serial
        self.s = serial.Serial(port, baud, timeout=0)
        self.last, self.t_last = None, 0.0

    def send(self, gesture):
        now = time.monotonic()
        if gesture != self.last or now - self.t_last > 0.5:
            self.s.write(f"G{GESTURE_IDS[gesture]}\n".encode())
            self.last, self.t_last = gesture, now


class App:
    def __init__(self, root, serial_port=None):
        self.root = root
        root.title("EMG-Controlled BiDexHand — Simulation")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{sw}x{sh}+0+0")
        try:
            root.attributes("-zoomed", True)
        except tk.TclError:
            pass
        self.VW = min(1920, max(640, int(sw * 0.55)) // 8 * 8)
        self.VH = min(1200, max(520, sh - 190) // 8 * 8)
        self.mode = tk.StringVar(value="manual")
        self.playing = False
        self.gesture = "rest"
        self.last_decision = None
        self.cam = {"azimuth": 30.0, "elevation": -15.0, "distance": 0.36}
        self.link = None
        if serial_port:
            try:
                self.link = SerialLink(serial_port)
            except Exception as e:
                print(f"serial port {serial_port} unavailable: {e}")
        self._build_sim()
        self._build_ml()
        self._build_ui()
        self.set_mode("ml" if self.ml else "manual")
        self.root.after(100, self.loop)

    # ------------------------------------------------------------------ setup
    def _build_sim(self):
        import mujoco
        self.mj = mujoco
        self.hand = BiDexHand()
        self.renderer = mujoco.Renderer(self.hand.m, height=self.VH, width=self.VW)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [-0.02, -0.01, 0.09]

    def _build_ml(self):
        self.ml, self.ml_error = None, None
        cand = HERE / "model"
        if (cand / "deploy.json").exists():
            try:
                self.ml = MLEngine(cand)
            except Exception as e:  # keep the demo usable in manual mode
                self.ml_error = f"{type(e).__name__}: {e}"
        self.recordings = sorted(glob.glob(str(HERE / "data" / "*.npz")))
        if self.ml and not self.recordings:
            self.ml_error, self.ml = "no recordings in simulation/data", None

    def _build_ui(self):
        st = ttk.Style()
        st.theme_use("clam")
        live = tk.Frame(self.root, bg=BG)
        live.pack(fill="both", expand=True, padx=8, pady=8)

        left = tk.Frame(live, bg=BG)
        left.pack(side="left", fill="y", padx=(4, 8))
        self.banner = tk.Label(left, text="", font=("Sans", 13, "bold"), fg="white", bg=BLUE, anchor="w", padx=10, pady=6)
        self.banner.pack(fill="x")
        self.view = tk.Label(left, bg="black")
        self.view.pack()
        cams = tk.Frame(left, bg=BG)
        cams.pack(fill="x", pady=4)
        for txt, az, el in [("Iso", 30, -15), ("Palm", 0, -10), ("Thumb side", 300, -10), ("Back", 180, -15), ("Top", 0, -75)]:
            tk.Button(cams, text=txt, command=lambda a=az, e=el: self.cam.update(azimuth=a, elevation=e)).pack(side="left", padx=2)
        tk.Button(cams, text="Zoom +", command=lambda: self.cam.update(distance=self.cam["distance"] * 0.85)).pack(side="left", padx=2)
        tk.Button(cams, text="Zoom −", command=lambda: self.cam.update(distance=self.cam["distance"] / 0.85)).pack(side="left", padx=2)
        self.view.bind("<B1-Motion>", self._drag)
        self.view.bind("<ButtonPress-1>", self._press)
        self.status = tk.Label(left, text="", font=("Mono", 9), bg=BG, fg=INK2, anchor="w", justify="left")
        self.status.pack(fill="x")

        right = tk.Frame(live, bg=PANEL, bd=1, relief="solid")
        right.pack(side="left", fill="both", expand=True)
        mrow = tk.Frame(right, bg=PANEL)
        mrow.pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(mrow, text="Mode:", font=("Sans", 12, "bold"), bg=PANEL).pack(side="left")
        self.rb_ml = tk.Radiobutton(mrow, text="ML mode (recorded EMG → CNN-LSTM)", variable=self.mode, value="ml",
                                    command=lambda: self.set_mode("ml"), bg=PANEL, font=("Sans", 11))
        self.rb_ml.pack(side="left", padx=6)
        tk.Radiobutton(mrow, text="Manual mode (no ML)", variable=self.mode, value="manual",
                       command=lambda: self.set_mode("manual"), bg=PANEL, font=("Sans", 11)).pack(side="left", padx=6)
        if not self.ml:
            self.rb_ml.configure(state="disabled")

        # ML controls
        self.mlf = tk.LabelFrame(right, text=" ML mode — EMG replay ", bg=PANEL, font=("Sans", 10, "bold"))
        self.mlf.pack(fill="x", padx=10, pady=4)
        r1 = tk.Frame(self.mlf, bg=PANEL)
        r1.pack(fill="x", pady=2)
        tk.Label(r1, text="Recording (unseen subject):", bg=PANEL).pack(side="left")
        self.rec_var = tk.StringVar(value=os.path.basename(self.recordings[0]) if self.recordings else "")
        ttk.Combobox(r1, textvariable=self.rec_var, values=[os.path.basename(r) for r in self.recordings],
                     width=28, state="readonly").pack(side="left", padx=4)
        self.play_btn = tk.Button(r1, text="▶ Play", width=8, command=self.toggle_play, bg=GREEN, fg="white")
        self.play_btn.pack(side="left", padx=4)
        tk.Button(r1, text="⟲ Restart", command=self.restart).pack(side="left")
        tk.Button(r1, text="⏭ Next gesture", command=self.next_gesture).pack(side="left", padx=4)
        self.skip_rest = tk.BooleanVar(value=True)
        tk.Checkbutton(r1, text="Skip long rest", variable=self.skip_rest, bg=PANEL).pack(side="left", padx=4)
        r2 = tk.Frame(self.mlf, bg=PANEL)
        r2.pack(fill="x", pady=2)
        tk.Label(r2, text="Command filter:", bg=PANEL).pack(side="left")
        self.filt_var = tk.StringVar(value="stable")
        for txt, val in [("Stable (hold 200 ms)", "stable"), ("Responsive (raw)", "raw"), ("Safe (conf ≥ 0.7)", "safe")]:
            tk.Radiobutton(r2, text=txt, variable=self.filt_var, value=val, command=self.apply_filter, bg=PANEL).pack(side="left")
        self.emg_canvas = tk.Canvas(self.mlf, height=100, bg="white", highlightthickness=0)
        self.emg_canvas.pack(fill="x", padx=4, pady=2)
        pr = tk.Frame(self.mlf, bg=PANEL)
        pr.pack(fill="x")
        self.pred_lbl = tk.Label(pr, text="Prediction: —", font=("Sans", 16, "bold"), bg=PANEL, fg=BLUE, anchor="w")
        self.pred_lbl.pack(side="left", padx=4)
        self.truth_lbl = tk.Label(pr, text="Ground truth: —", font=("Sans", 12), bg=PANEL, fg=INK2)
        self.truth_lbl.pack(side="right", padx=4)
        self.prob_canvas = tk.Canvas(self.mlf, height=150, bg="white", highlightthickness=0)
        self.prob_canvas.pack(fill="x", padx=4, pady=2)
        self.filter_lbl = tk.Label(self.mlf, text="", bg=PANEL, fg=INK2, anchor="w", justify="left", font=("Sans", 9))
        self.filter_lbl.pack(fill="x", padx=4)

        # Manual controls
        self.manf = tk.LabelFrame(right, text=" Manual mode — presenter selects gesture (no ML) ", bg=PANEL, font=("Sans", 10, "bold"))
        self.manf.pack(fill="x", padx=10, pady=4)
        grid = tk.Frame(self.manf, bg=PANEL)
        grid.pack(fill="x", pady=2)
        labels = {"rest": "Rest / Open", "fist": "Fist (power)", "flexion": "Close hand", "extension": "Open wide",
                  "pinch_index": "Pinch (index)", "pinch_middle": "Tripod", "pinch_ring": "Thumb–ring",
                  "pinch_little": "Thumb–little", "index_point": "Index point", "peace": "V sign"}
        for k, g in enumerate(ALL_POSES):
            tk.Button(grid, text=labels[g], width=13, command=lambda g=g: self.manual(g)).grid(row=k // 5, column=k % 5, padx=2, pady=2)

        # Servo panel
        cf = tk.LabelFrame(right, text=" Microcontroller output → 16 servos (PCA9685, 50 Hz PWM) ", bg=PANEL, font=("Sans", 10, "bold"))
        cf.pack(fill="both", expand=True, padx=10, pady=4)
        self.cmd_lbl = tk.Label(cf, text="", font=("Sans", 12, "bold"), bg=PANEL, fg=INK, anchor="w")
        self.cmd_lbl.pack(fill="x", padx=4)
        self.act_canvas = tk.Canvas(cf, height=16 * 15 + 8, bg="white", highlightthickness=0)
        self.act_canvas.pack(fill="x", padx=4, pady=2)
        tk.Label(cf, text="Bars: blue = commanded angle, grey = simulated servo angle (speed-limited 0.10 s/60°). "
                          "Text = angle and PWM pulse width sent by the MCU.",
                 bg=PANEL, fg=GREY, font=("Sans", 8), wraplength=520, justify="left").pack(fill="x", padx=4)

    # ------------------------------------------------------------------ interaction
    def _press(self, e):
        self._xy = (e.x, e.y)

    def _drag(self, e):
        dx, dy = e.x - self._xy[0], e.y - self._xy[1]
        self._xy = (e.x, e.y)
        self.cam["azimuth"] -= dx * 0.4
        self.cam["elevation"] = float(np.clip(self.cam["elevation"] - dy * 0.4, -89, 89))

    def set_mode(self, m):
        self.mode.set(m)
        if m == "ml" and self.ml:
            self.banner.configure(text="ML MODE — CNN-LSTM classifying RECORDED EMG of an UNSEEN subject", bg=BLUE)
            if self.ml.rec is None:
                self.restart()
        else:
            self.playing = False
            self.play_btn.configure(text="▶ Play", bg=GREEN)
            self.banner.configure(text="MANUAL MODE — gesture chosen by presenter. No ML prediction involved.", bg=ORANGE)
            if not self.ml:
                self.banner.configure(text=f"MANUAL MODE — ML unavailable ({self.ml_error or 'model not found'})", bg=ORANGE)

    def manual(self, g):
        if self.mode.get() != "manual":
            self.set_mode("manual")
        self.gesture = g

    def restart(self):
        if not self.ml:
            return
        path = next((r for r in self.recordings if os.path.basename(r) == self.rec_var.get()), self.recordings[0])
        self.ml.load(path)
        self.apply_filter()
        self.emg_hist = np.zeros((int(3.0 * 1000 / STEP_MS), 6))

    def apply_filter(self):
        """Switch the decoder's command filter (held-out numbers for each are shown in the panel)."""
        if not (self.ml and self.ml.rec):
            return
        dec, pp = self.ml.dec, self.ml.dep["postproc"]
        if self.filt_var.get() == "raw":
            dec.thr, dec.smooth, dec.k = 0.0, "none", 1
        elif self.filt_var.get() == "stable":   # switch only after 4 consecutive equal decisions (200 ms)
            dec.thr, dec.smooth, dec.k = 0.0, "debounce", 4
            dec.cur, dec.cand, dec.cnt = 0, None, 0
        else:
            dec.thr, dec.smooth, dec.k = pp.get("conf_threshold", 0.0), pp.get("smooth", "none"), int(pp.get("smooth_len", 1))
        dec.hist = collections.deque(maxlen=max(dec.k, 1))
        dec.ema = None

    def toggle_play(self):
        if self.mode.get() != "ml":
            self.set_mode("ml")
        if self.ml and (self.ml.rec is None or os.path.basename(self.ml.rec) != self.rec_var.get()):
            self.restart()
        self.playing = not self.playing
        self.play_btn.configure(text="⏸ Pause" if self.playing else "▶ Play", bg=RED if self.playing else GREEN)

    def next_gesture(self):
        """Jump to 0.6 s before the next gesture; the skipped EMG is still processed by the model."""
        if not (self.ml and self.ml.rec):
            return
        on = self.ml.next_onset(self.ml.pos + int(0.5 * self.ml.dep["fs"]))
        if on is not None:
            d = self.ml.fast_forward(on - int(0.6 * self.ml.dep["fs"]))
            if d:
                self.last_decision, self.gesture = d, d["name"]

    def skip(self, s):
        if self.ml and self.ml.rec:
            for _ in range(int(s * 1000 / STEP_MS)):
                if self.ml.tick() is None:
                    break

    # ------------------------------------------------------------------ main loop
    def loop(self):
        t0 = time.perf_counter()
        ml_ms = 0.0
        if self.mode.get() == "ml" and self.ml and self.playing:
            a = time.perf_counter()
            fs = self.ml.dep["fs"]
            if self.skip_rest.get() and self.ml.gt[min(self.ml.pos, len(self.ml.gt) - 1)] <= 0:
                on = self.ml.next_onset(self.ml.pos)
                if on is not None and on - self.ml.pos > int(1.2 * fs):
                    self.ml.fast_forward(on - int(0.6 * fs))
            d = self.ml.tick()
            ml_ms = (time.perf_counter() - a) * 1000
            if d is None:
                self.playing = False
                self.play_btn.configure(text="▶ Play", bg=GREEN)
            elif not d.get("warming"):
                self.last_decision = d
                self.gesture = d["name"]
            seg = self.ml.emg[max(0, self.ml.pos - 52):self.ml.pos]
            if len(seg):
                ch = [0, 4, 8, 12, 16, 20]
                rms = np.sqrt((seg[:, ch] ** 2).mean(0)) / np.array(self.ml.dep["scale"])[ch]
                self.emg_hist = np.vstack([self.emg_hist[1:], rms])
        self.hand.command(ALL_POSES[self.gesture]["servo"])
        if self.link:
            self.link.send(self.gesture)
        self.hand.step(STEP_MS / 1000)
        sim_ms = (time.perf_counter() - t0) * 1000 - ml_ms
        self.render()
        self.draw_panels(ml_ms, sim_ms)
        spent = (time.perf_counter() - t0) * 1000
        self.root.after(max(1, int(STEP_MS - spent)), self.loop)

    def render(self):
        c = self.camera
        c.azimuth, c.elevation, c.distance = self.cam["azimuth"], self.cam["elevation"], self.cam["distance"]
        self.renderer.update_scene(self.hand.d, camera=c)
        img = Image.fromarray(self.renderer.render())
        dr = ImageDraw.Draw(img)
        fn = ALL_POSES[self.gesture]["function"]
        dr.rectangle([0, 0, img.width, 34], fill=(20, 24, 30))
        dr.text((10, 8), f"BiDexHand · 16 servos · command: {self.gesture.upper()}  ({fn})", fill=(255, 255, 255), font=FONT)
        dr.text((10, img.height - 22), f"sim t = {self.hand.time:6.2f} s   [SIMULATION]", fill=(220, 220, 220), font=FONT_S)
        self.tkimg = ImageTk.PhotoImage(img)
        self.view.configure(image=self.tkimg)

    def draw_panels(self, ml_ms, sim_ms):
        fn = ALL_POSES[self.gesture]["function"]
        src = "ML decision" if self.mode.get() == "ml" else "MANUAL selection"
        self.cmd_lbl.configure(text=f"{src}: {self.gesture} (id {GESTURE_IDS[self.gesture]})  →  {fn}")
        cv = self.act_canvas
        cv.delete("all")
        W = max(cv.winfo_width(), 600)
        pwm = self.hand.servo_pwm_us()
        x0, x1 = 120, W - 190
        xm = (x0 + x1) / 2
        for i in range(16):
            y = 4 + i * 15
            cv.create_text(4, y + 6, text=SERVO_NAMES[i], anchor="w", font=("Sans", 8))
            cv.create_rectangle(x0, y, x1, y + 12, outline="#ddd")
            cv.create_line(xm, y, xm, y + 12, fill="#ccc")
            t, a = self.hand.target[i] / 90, self.hand.servo[i] / 90
            cv.create_rectangle(xm, y, xm + (x1 - xm) * t, y + 5, fill=BLUE, outline="")
            cv.create_rectangle(xm, y + 6, xm + (x1 - xm) * a, y + 12, fill="#9a9890", outline="")
            cv.create_text(x1 + 6, y + 6, anchor="w", font=("Mono", 8),
                           text=f"ch{i:02d} {self.hand.servo[i]:+6.1f}° {pwm[i]:5.0f} µs")
        link = f" | serial: {self.link.s.port}" if self.link else ""
        self.status.configure(text=f"loop: ML {ml_ms:4.1f} ms | servo+kinematics {sim_ms:5.1f} ms per {STEP_MS} ms tick | "
                                   f"total servo travel {self.hand.travel_deg / 1000:.1f}k°{link}")
        if self.mode.get() == "ml" and self.ml and self.ml.rec:
            ml = self.ml
            self.mlf.configure(text=f" ML mode — subject S{ml.subject:02d}, session {ml.session}, "
                                    f"t = {ml.pos / ml.dep['fs']:6.1f} / {len(ml.emg) / ml.dep['fs']:.0f} s ")
            cv = self.emg_canvas
            cv.delete("all")
            W = max(cv.winfo_width(), 600)
            H = 100
            h = self.emg_hist
            for k in range(h.shape[1]):
                pts = []
                for j, v in enumerate(h[:, k]):
                    pts += [j * W / len(h), H - 6 - min(v, 4) / 4 * (H - 12) * 0.9]
                cv.create_line(*pts, fill=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"][k], width=1.2)
            cv.create_text(6, 8, anchor="w", text="EMG envelope, 6 of 24 channels (last 3 s, normalised)", font=("Sans", 8), fill=GREY)
            d = self.last_decision
            if d:
                ok = d["name"] == d["truth"]
                self.pred_lbl.configure(text=f"Prediction: {d['name']}  ({d['conf'] * 100:.0f}%)", fg=BLUE)
                self.truth_lbl.configure(text=f"Ground truth: {d['truth']}  {'✓' if ok else '✗'}", fg=GREEN if ok else RED)
                cv = self.prob_canvas
                cv.delete("all")
                W = max(cv.winfo_width(), 600)
                for i, (n, p) in enumerate(zip(ml.names, d["probs"])):
                    y = 6 + i * 18
                    cv.create_text(6, y + 7, text=n, anchor="w", font=("Sans", 9))
                    cv.create_rectangle(120, y, 120 + (W - 200) * float(p), y + 13, fill=BLUE if n == d["name"] else "#b4b2aa", outline="")
                    cv.create_text(W - 70, y + 7, text=f"{p * 100:5.1f}%", anchor="w", font=("Mono", 9))
                meas = {"safe": "held-out: 1.1 false triggers/min, 49 % gestures correct",
                        "stable": "held-out: 2.7 false triggers/min, 72 % gestures correct",
                        "raw": "held-out: 8.1 false triggers/min, 72 % gestures correct"}[self.filt_var.get()]
                acc = ml.n_ok / max(ml.n_tot, 1)
                self.filter_lbl.configure(text=(
                    f"Raw argmax: {d['raw']} → filter ({ml.dec.smooth}, conf ≥ {ml.dec.thr:.2f}) → decision: {d['name']}   [{meas}]\n"
                    f"Window {ml.dep['window_ms']} ms, step {ml.dep['step_ms']} ms, inference {np.mean(ml.dec.compute_ms[-20:]):.2f} ms. "
                    f"Agreement with ground truth so far: {acc * 100:.1f}% of {ml.n_tot} decisions "
                    f"(includes transitions; S{ml.subject:02d} never used in training)."))

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass


def selftest(app, root, seconds):
    """Scripted run: manual gestures, then ML playback; saves frames; exits non-zero on error."""
    out = HERE / "selftest"
    out.mkdir(exist_ok=True)
    plan = [("manual", g) for g in ("fist", "pinch_index", "pinch_middle", "rest")] + [("ml", None)]
    per = max(1.5, seconds / len(plan))
    state = {"k": -1, "t": 0.0, "log": []}

    def step():
        try:
            k = int(state["t"] // per)
            if k != state["k"]:
                if state["k"] >= 0:   # save the frame reached at the end of the previous phase
                    m, g = plan[state["k"]]
                    ImageTk.getimage(app.tkimg).convert("RGB").save(out / f"{state['k']:02d}_{m}_{g or 'replay'}.png")
                    state["log"].append(f"{m}:{g or app.gesture} servo={np.round(app.hand.servo).astype(int).tolist()}")
                if k >= len(plan):
                    if app.ml and app.ml.n_tot:
                        state["log"].append(f"ml decisions={app.ml.n_tot} agree={app.ml.n_ok / app.ml.n_tot:.3f}")
                    (out / "selftest.log").write_text("\n".join(state["log"]) + "\nSELFTEST OK\n")
                    print("\n".join(state["log"]))
                    print("SELFTEST OK")
                    app.close()
                    root.destroy()
                    return
                state["k"] = k
                m, g = plan[k]
                if m == "manual":
                    app.manual(g)
                elif app.ml:
                    app.set_mode("ml")
                    app.skip(20)
                    app.toggle_play()
            state["t"] += 0.25
            root.after(250, step)
        except Exception:
            import traceback
            traceback.print_exc()
            root.destroy()
            os._exit(2)
    root.after(1500, step)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", type=float, default=0, help="run a scripted self-test for N seconds and exit")
    ap.add_argument("--serial", default=None, help="serial port of the servo microcontroller, e.g. /dev/ttyUSB0")
    a = ap.parse_args()
    root = tk.Tk()
    try:
        app = App(root, a.serial)
        if a.selftest:
            selftest(app, root, a.selftest)
    except Exception as e:
        import traceback
        traceback.print_exc()
        if not a.selftest:
            messagebox.showerror("EMG BiDexHand simulation", f"Startup failed:\n{type(e).__name__}: {e}")
        raise
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
