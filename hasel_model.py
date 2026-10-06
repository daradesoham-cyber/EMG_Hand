"""
HASEL Actuator phenomenological model module.
Implements the quadratic voltage command and linear stroke-decay electrohydraulic behavior:
Force = Fmax * (Voltage_Command^2) * (1 - Current_Stroke / Max_Stroke)
Supports drop-in replacement with experimental empirical lookups.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from parameters import HaselActuatorConfig

@dataclass
class HaselActuatorState:
    name: str
    voltage_command: float  # [0.0, 1.0]
    current_stroke: float   # [m]
    output_force: float     # [N]
    is_saturated: bool
    is_stroke_limited: bool

class HaselActuatorModel:
    """
    Phenomenological model of an electrohydraulic HASEL pouch stack.
    Deterministic, physics-inspired, and designed with a swappable interface.
    """
    def __init__(self, config: HaselActuatorConfig):
        self.config = config
        self.fmax = float(config.fmax)
        self.max_stroke = float(config.max_stroke)
        self.name = config.name

    def compute_force(self, voltage_command: float, current_stroke: float) -> HaselActuatorState:
        """
        Calculates instantaneous tension force provided by the HASEL actuator.
        
        Args:
            voltage_command: Normalized voltage control signal in [0.0, 1.0]
            current_stroke: Actuator contraction displacement in meters [0.0, max_stroke]
            
        Returns:
            HaselActuatorState with computed tension force and state flags.
        """
        # Clamp inputs within valid operational domains
        v_clamped = float(np.clip(voltage_command, 0.0, 1.0))
        stroke_clamped = float(np.clip(current_stroke, 0.0, self.max_stroke))
        
        stroke_limited = (current_stroke >= self.max_stroke) or (current_stroke <= 0.0)
        
        # Phenomenological force relationship:
        # Electrostatic pressure is proportional to V^2 (Maxwell stress).
        # Mechanical contraction reduces hydraulic pouch expansion headroom, yielding linear force drop.
        force_unclamped = self.fmax * (v_clamped ** 2) * (1.0 - (stroke_clamped / self.max_stroke))
        
        force = float(np.clip(force_unclamped, 0.0, self.fmax))
        saturated = (force >= self.fmax)
        
        return HaselActuatorState(
            name=self.name,
            voltage_command=v_clamped,
            current_stroke=stroke_clamped,
            output_force=force,
            is_saturated=saturated,
            is_stroke_limited=stroke_limited
        )

    def calculate_mass_and_volume(self) -> Dict[str, float]:
        """Returns physical packaging volume and mass."""
        vol = self.config.volume_m3
        mass = self.config.mass_kg
        return {
            "volume_m3": vol,
            "volume_cm3": vol * 1e6,
            "mass_kg": mass,
            "mass_g": mass * 1e3
        }

class SixHaselSystem:
    """Coordinates all six independent HASEL actuator channels."""
    def __init__(self, actuator_configs: Dict[str, HaselActuatorConfig]):
        self.actuators = {
            name: HaselActuatorModel(cfg)
            for name, cfg in actuator_configs.items()
        }

    def compute_all_forces(self, voltages: Dict[str, float], strokes: Dict[str, float]) -> Dict[str, HaselActuatorState]:
        states = {}
        for name, act in self.actuators.items():
            v = voltages.get(name, 0.0)
            s = strokes.get(name, 0.0)
            states[name] = act.compute_force(v, s)
        return states

    def total_mass_kg(self) -> float:
        return sum(act.config.mass_kg for act in self.actuators.values())

    def total_volume_m3(self) -> float:
        return sum(act.config.volume_m3 for act in self.actuators.values())
