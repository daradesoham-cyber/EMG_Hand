"""BiDexHand (16-servo anthropomorphic hand) simulated in MuJoCo.

The hand geometry is the same URDF + STL meshes used by the ROS 2 / RViz demo
(`hand_rviz`). The 16 servo angles use the same index order as the
`/hand_servo_input` topic, and the 16 -> 21 joint mapping (PIP compensation,
DIP from PIP through the antiparallelogram linkage) is copied from
`hand_rviz/joint_state_forwarding_node.py`, so a pose shown here looks the same
as it does in RViz.

Servo dynamics: every servo is a first-order position loop with a speed limit
(SG90/MG90S class servo, about 0.10 s per 60 deg at 5 V), stepped at 1 kHz,
which is what a PWM servo driven by the microcontroller does.
"""
import math
import re
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "bidexhand"

SERVO_NAMES = [
    "Thumb CMC abd", "Index MCP abd", "Middle MCP abd", "Ring MCP abd", "Little MCP abd",
    "Thumb MCP abd", "Index MCP flex", "Middle MCP flex", "Ring MCP flex", "Little MCP flex",
    "Thumb MCP flex", "Index PIP flex", "Middle PIP flex", "Ring PIP flex", "Little PIP flex",
    "Thumb CMC flex",
]
JOINT_NAMES = [
    "tcf_joint", "tca_joint", "tma_joint", "tmf_joint", "tdf_joint",
    "ima_joint", "imf_joint", "ipf_joint", "idf_joint",
    "mma_joint", "mmf_joint", "mpf_joint", "mdf_joint",
    "rma_joint", "rmf_joint", "rpf_joint", "rdf_joint",
    "pma_joint", "pmf_joint", "ppf_joint", "pdf_joint",
]


def _pose(thumb_cmc_abd=0, thumb_cmc_flex=0, thumb_mcp_abd=0, thumb_mcp_flex=0,
          index=(0, 0), middle=(0, 0), ring=(0, 0), little=(0, 0), spread=(0, 0, 0, 0)):
    """Build a 16-servo command (degrees) from per-finger (MCP flex, PIP flex) values."""
    s = [0.0] * 16
    s[0], s[15], s[5], s[10] = thumb_cmc_abd, thumb_cmc_flex, thumb_mcp_abd, thumb_mcp_flex
    s[1:5] = spread
    for k, (mcp, pip) in enumerate((index, middle, ring, little)):
        s[6 + k], s[11 + k] = mcp, pip
    return s


# Gesture (class name of the CNN-LSTM) -> 16 servo angles in degrees, and the hand function it drives.
# Pinch poses: thumb angles solved numerically (Nelder-Mead) so the thumb tip meets the target fingertip.
GESTURES = {
    "rest":         {"servo": _pose(), "function": "Open / relaxed"},
    "fist":         {"servo": _pose(20, -60, 0, 50, (80, 85), (80, 85), (80, 85), (80, 85)),
                     "function": "Power grasp (all fingers closed)"},
    "flexion":      {"servo": _pose(10, -40, 0, 30, (60, 60), (60, 60), (60, 60), (60, 60)),
                     "function": "Close hand (wrist-flexion trigger)"},
    "extension":    {"servo": _pose(0, 0, 0, 0, (-5, 0), (-5, 0), (-5, 0), (-5, 0), (-8, -3, 3, 8)),
                     "function": "Open hand wide (wrist-extension trigger)"},
    "pinch_index":  {"servo": _pose(-1, -30, 8, -20, (45, 45)),
                     "function": "Precision pinch (thumb + index)"},
    "pinch_middle": {"servo": _pose(0, -40, 0, -11, (45, 50), (50, 55)),
                     "function": "Tripod (thumb + index + middle)"},
    "pinch_ring":   {"servo": _pose(0, -59, -8, 0, (0, 0), (0, 0), (55, 60)),
                     "function": "Thumb-ring pinch"},
    "pinch_little": {"servo": _pose(0, -70, -24, 0, (0, 0), (0, 0), (0, 0), (60, 65)),
                     "function": "Thumb-little pinch"},
}
EXTRA_PRESETS = {  # manual-only presets (not classes of the model)
    "index_point": {"servo": _pose(20, -60, 0, 50, (0, 0), (80, 85), (80, 85), (80, 85)),
                    "function": "Index point (manual preset)"},
    "peace":       {"servo": _pose(20, -60, 0, 50, (0, 0), (0, 0), (80, 85), (80, 85), (-8, 8, 0, 0)),
                    "function": "Two-finger 'V' (manual preset)"},
}
ALL_POSES = {**GESTURES, **EXTRA_PRESETS}


def dip_from_pip(pip):
    """DIP angle from PIP angle, antiparallelogram linkage (same as hand_rviz)."""
    neutral_pip, neutral_dip = 1.968, 0.3978
    return 2 * math.atan(0.87 / 2.87 / math.tan((neutral_pip - pip) / 2)) - neutral_dip


def servo_to_joints(servo_deg):
    """16 servo angles (deg, /hand_servo_input order) -> 21 URDF joint angles (rad, JOINT_NAMES order).
    Identical maths to hand_rviz/joint_state_forwarding_node.py."""
    r = [math.radians(a) for a in servo_deg]
    m = [0.0] * 21
    m[0], m[1], m[2], m[3] = r[15], r[0], r[5], r[10]
    m[5], m[6], m[7] = r[1], r[6], r[11]
    m[9], m[10], m[11] = r[2], r[7], r[12]
    m[13], m[14], m[15] = r[3], r[8], r[13]
    m[17], m[18], m[19] = r[4], r[9], r[14]
    for p, mcp in ((7, 6), (11, 10), (15, 14), (19, 18)):
        m[p] -= m[mcp] * 0.7
    m[3] = m[3] + m[2] * 0.7 + 1
    for d, p in ((4, 3), (8, 7), (12, 11), (16, 15), (20, 19)):
        m[d] = dip_from_pip(m[p])
    for p in (3, 7, 11, 15, 19):
        m[p] *= 0.7
    m[1] *= 0.5
    return np.array(m)


def _mjcf_from_urdf():
    s = (ASSET_DIR / "hand_urdf.urdf").read_text().replace("package://hand_rviz/urdf/meshes/", "")
    compiler = (f'<mujoco><compiler meshdir="{ASSET_DIR / "meshes"}" balanceinertia="true" '
                f'discardvisual="false" fusestatic="false"/></mujoco>')
    s = re.sub(r"(<robot[^>]*>)", r"\1" + compiler, s, count=1)
    m = mujoco.MjModel.from_xml_string(s)
    spec_xml = Path("/dev/shm" if Path("/dev/shm").exists() else "/tmp") / "bidexhand_tmp.xml"
    mujoco.mj_saveLastXML(str(spec_xml), m)
    xml = spec_xml.read_text()
    spec_xml.unlink(missing_ok=True)
    # scene: lights, floor, skybox, offscreen buffer
    xml = xml.replace("<worldbody>", """<worldbody>
    <light directional="true" pos="0.3 -0.5 0.8" dir="-0.3 0.5 -0.8" diffuse="0.85 0.85 0.85"/>
    <light directional="true" pos="-0.3 0.4 0.6" dir="0.3 -0.4 -0.6" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="1 1 0.01" pos="0 0 -0.12" material="grid" contype="0" conaffinity="0"/>""", 1)
    assets = """<asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.16 0.18 0.22" rgb2="0.2 0.22 0.27"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <material name="shell" rgba="0.82 0.84 0.88 1" specular="0.4" shininess="0.5"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.32 0.36 0.42" rgb2="0.08 0.09 0.11" width="256" height="256"/>
  </asset>"""
    xml = xml.replace("<asset>", assets.rsplit("</asset>", 1)[0], 1) if "<asset>" in xml else xml.replace("<worldbody>", assets + "<worldbody>", 1)
    xml = xml.replace("</mujoco>", '<visual><global offwidth="1920" offheight="1200"/><quality shadowsize="4096"/></visual>\n</mujoco>', 1)
    return xml


class BiDexHand:
    """Kinematic BiDexHand driven by 16 simulated hobby servos."""

    SERVO_SPEED_DPS = 600.0      # 0.10 s / 60 deg
    SERVO_TAU = 0.04             # position-loop time constant (s)

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_string(_mjcf_from_urdf())
        self.d = mujoco.MjData(self.m)
        shell = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_MATERIAL, "shell")
        for g in range(self.m.ngeom):
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                self.m.geom_matid[g] = shell
                self.m.geom_rgba[g] = [0.82, 0.84, 0.88, 1]
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINT_NAMES])
        self.lo = np.array([self.m.jnt_range[self.m.joint(n).id][0] for n in JOINT_NAMES])
        self.hi = np.array([self.m.jnt_range[self.m.joint(n).id][1] for n in JOINT_NAMES])
        self.target = np.zeros(16)       # commanded servo angles (deg)
        self.servo = np.zeros(16)        # actual servo angles (deg)
        self.time = 0.0
        self.travel_deg = 0.0
        self._apply()

    def command(self, servo_deg):
        self.target = np.asarray(servo_deg, float)

    def step(self, dt_total, dt=0.001):
        """Advance the servo model by dt_total seconds in dt steps, then update the kinematics."""
        n = max(1, int(round(dt_total / dt)))
        vmax = self.SERVO_SPEED_DPS * dt
        for _ in range(n):
            dv = np.clip((self.target - self.servo) * dt / self.SERVO_TAU, -vmax, vmax)
            self.servo += dv
            self.travel_deg += float(np.abs(dv).sum())
        self.time += n * dt
        self._apply()

    def _apply(self):
        q = np.clip(servo_to_joints(self.servo), self.lo, self.hi)
        self.d.qpos[self.qadr] = q
        self.d.time = self.time
        mujoco.mj_forward(self.m, self.d)

    def settled(self, tol=1.0):
        return float(np.abs(self.target - self.servo).max()) < tol

    def servo_pwm_us(self):
        """Servo angle -> PWM pulse width the MCU would output (1500 us = 0 deg, +-90 deg = +-1000 us)."""
        return 1500 + np.clip(self.servo, -90, 90) / 90 * 1000
