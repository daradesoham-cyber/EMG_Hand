"""
Parameters module for the Parameterized Six-HASEL Biomechatronic Prosthetic Hand.
Defines explicit, typed, and serializable parameter structures for all design variables.
Classifies all parameters with provenance:
LITERATURE_REFERENCE | DESIGN_TARGET | SIMULATION_ASSUMPTION | EXPERIMENTALLY_VALIDATED.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

@dataclass
class HaselActuatorConfig:
    name: str
    fmax: float  # [N] Design target maximum force
    max_stroke: float  # [m] Maximum stroke
    resting_length: float  # [m]
    width: float  # [m]
    thickness_per_layer: float  # [m]
    num_layers: int
    density: float  # [kg/m^3]
    position: List[float]  # [x, y, z] relative mounting position [m]
    orientation: List[float]  # [roll, pitch, yaw] in radians
    classification: str = "DESIGN_TARGET"

    @property
    def volume_m3(self) -> float:
        """Calculate packaging volume of the deformable pouch stack."""
        return self.resting_length * self.width * (self.thickness_per_layer * self.num_layers)

    @property
    def mass_kg(self) -> float:
        """Calculate actuator mass based on volumetric density."""
        return self.volume_m3 * self.density

@dataclass
class PulleyGeometry:
    mcp_radius: float  # [m]
    pip_radius: float  # [m]
    dip_radius: float  # [m]
    eccentric_cam_offset: float = 0.0  # [m]
    tendon_moment_arm_bias: float = 1.0

@dataclass
class FingerSegmentGeometry:
    proximal_len: float  # [m]
    middle_len: float  # [m]
    distal_len: float  # [m]
    width_proximal: float  # [m]
    width_distal: float  # [m]
    mcp_stiffness: float = 0.15  # [N*m/rad] Passive flexure stiffness
    mcp_damping: float = 0.02    # [N*m*s/rad]
    pip_stiffness: float = 0.15
    pip_damping: float = 0.02
    dip_stiffness: float = 0.15
    dip_damping: float = 0.02
    mcp_range: List[float] = field(default_factory=lambda: [0.0, 1.5708])  # [rad] (0 - 90 deg)
    pip_range: List[float] = field(default_factory=lambda: [0.0, 1.9198])  # [rad] (0 - 110 deg)
    dip_range: List[float] = field(default_factory=lambda: [0.0, 1.3963])  # [rad] (0 - 80 deg)

@dataclass
class ThumbGeometry:
    metacarpal_len: float  # [m]
    proximal_len: float    # [m]
    distal_len: float      # [m]
    width: float           # [m]
    cmc_pos: List[float]   # [x, y, z] on palm
    cmc_axis_euler: List[float]  # [roll, pitch, yaw] orientation of opposition axis
    cmc_stiffness: float = 0.18
    cmc_damping: float = 0.025
    mp_stiffness: float = 0.15
    mp_damping: float = 0.02
    ip_stiffness: float = 0.15
    ip_damping: float = 0.02
    cmc_range: List[float] = field(default_factory=lambda: [0.0, 1.0472])  # 0 - 60 deg
    flex_range: List[float] = field(default_factory=lambda: [0.0, 1.5708]) # 0 - 90 deg

@dataclass
class PalmGeometry:
    length: float  # [m] (carpal-to-MCP)
    width: float   # [m]
    thickness: float # [m]
    finger_spacing: float # [m] MCP transverse center-to-center spacing
    mass: float = 0.220 # [kg] Estimated 3D printed chassis mass

@dataclass
class WristForearmGeometry:
    wrist_flexion_range: List[float] = field(default_factory=lambda: [-0.7854, 0.7854]) # +/- 45 deg
    wrist_deviation_range: List[float] = field(default_factory=lambda: [-0.349, 0.349]) # +/- 20 deg
    forearm_length: float = 0.220 # [m]
    forearm_cross_section_area: float = 0.0055 # [m^2] (approx 75mm x 75mm)
    forearm_max_volume: float = 0.0012 # [m^3] ~ 1.2 Liters

@dataclass
class HandParameters:
    palm: PalmGeometry
    fingers: Dict[str, FingerSegmentGeometry]
    thumb: ThumbGeometry
    pulleys: Dict[str, PulleyGeometry]
    actuators: Dict[str, HaselActuatorConfig]
    wrist: WristForearmGeometry
    placement_category: str = "forearm" # forearm | palm | hybrid

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

def load_default_parameters() -> HandParameters:
    """Load default anthropometric literature values & design targets."""
    # Palm
    palm = PalmGeometry(
        length=0.105,
        width=0.085,
        thickness=0.026,
        finger_spacing=0.019,
        mass=0.210
    )

    # 4 digits
    fingers = {
        "index": FingerSegmentGeometry(
            proximal_len=0.043, middle_len=0.025, distal_len=0.020,
            width_proximal=0.019, width_distal=0.015
        ),
        "middle": FingerSegmentGeometry(
            proximal_len=0.048, middle_len=0.028, distal_len=0.022,
            width_proximal=0.020, width_distal=0.016
        ),
        "ring": FingerSegmentGeometry(
            proximal_len=0.042, middle_len=0.026, distal_len=0.020,
            width_proximal=0.018, width_distal=0.0145
        ),
        "little": FingerSegmentGeometry(
            proximal_len=0.034, middle_len=0.019, distal_len=0.017,
            width_proximal=0.016, width_distal=0.013
        )
    }

    # Thumb
    thumb = ThumbGeometry(
        metacarpal_len=0.046,
        proximal_len=0.032,
        distal_len=0.022,
        width=0.021,
        cmc_pos=[0.022, -0.028, -0.015],
        cmc_axis_euler=[0.65, 0.45, 0.35]
    )

    # Pulley radii default 8mm (MCP), 7mm (PIP), 6mm (DIP)
    pulleys = {
        "index": PulleyGeometry(mcp_radius=0.009, pip_radius=0.007, dip_radius=0.006),
        "middle": PulleyGeometry(mcp_radius=0.009, pip_radius=0.007, dip_radius=0.006),
        "ring": PulleyGeometry(mcp_radius=0.008, pip_radius=0.007, dip_radius=0.006),
        "little": PulleyGeometry(mcp_radius=0.008, pip_radius=0.006, dip_radius=0.005),
        "thumb_flex": PulleyGeometry(mcp_radius=0.009, pip_radius=0.007, dip_radius=0.006),
        "thumb_opp": PulleyGeometry(mcp_radius=0.012, pip_radius=0.000, dip_radius=0.000)
    }

    # Six HASEL actuators
    # Actuators 1-4: Primary fingers (Fmax=375N, stroke=40.1mm)
    # Actuators 5-6: Thumb flexion & opposition (Fmax=100N, stroke=28mm)
    actuators = {
        "index_hasel": HaselActuatorConfig(
            name="index_hasel", fmax=375.0, max_stroke=0.0401, resting_length=0.120,
            width=0.032, thickness_per_layer=0.0018, num_layers=8, density=1100.0,
            position=[0.018, 0.0, -0.150], orientation=[0.0, 0.0, 0.0]
        ),
        "middle_hasel": HaselActuatorConfig(
            name="middle_hasel", fmax=375.0, max_stroke=0.0401, resting_length=0.120,
            width=0.032, thickness_per_layer=0.0018, num_layers=8, density=1100.0,
            position=[0.006, 0.0, -0.150], orientation=[0.0, 0.0, 0.0]
        ),
        "ring_hasel": HaselActuatorConfig(
            name="ring_hasel", fmax=375.0, max_stroke=0.0401, resting_length=0.120,
            width=0.032, thickness_per_layer=0.0018, num_layers=8, density=1100.0,
            position=[-0.006, 0.0, -0.150], orientation=[0.0, 0.0, 0.0]
        ),
        "little_hasel": HaselActuatorConfig(
            name="little_hasel", fmax=375.0, max_stroke=0.0401, resting_length=0.120,
            width=0.032, thickness_per_layer=0.0018, num_layers=8, density=1100.0,
            position=[-0.018, 0.0, -0.150], orientation=[0.0, 0.0, 0.0]
        ),
        "thumb_flex_hasel": HaselActuatorConfig(
            name="thumb_flex_hasel", fmax=100.0, max_stroke=0.0280, resting_length=0.090,
            width=0.024, thickness_per_layer=0.0018, num_layers=6, density=1100.0,
            position=[0.025, 0.012, -0.120], orientation=[0.1, 0.2, 0.0]
        ),
        "thumb_opp_hasel": HaselActuatorConfig(
            name="thumb_opp_hasel", fmax=100.0, max_stroke=0.0280, resting_length=0.090,
            width=0.024, thickness_per_layer=0.0018, num_layers=6, density=1100.0,
            position=[0.030, -0.012, -0.120], orientation=[-0.1, 0.3, 0.2]
        )
    }

    wrist = WristForearmGeometry()

    return HandParameters(
        palm=palm,
        fingers=fingers,
        thumb=thumb,
        pulleys=pulleys,
        actuators=actuators,
        wrist=wrist,
        placement_category="forearm"
    )
