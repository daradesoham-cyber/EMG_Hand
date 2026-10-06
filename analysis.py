"""
Kinematic and dynamic performance analysis module.
Computes:
- Joint ranges of motion (MCP, PIP, DIP, Thumb CMC, Thumb Flexion)
- Tendon moment arms and mechanical advantage
- Force transmission efficiency: eta = useful fingertip force / actuator force
- Hand mass, actuator volume, forearm volume, center of mass (COM)
- Static wrist gravity moment / reaction torque
"""

from typing import Dict, Any, Tuple
import numpy as np
import mujoco
from parameters import HandParameters
from simulation import generate_hand_mjcf

class MechanicalAnalyzer:
    def __init__(self, params: HandParameters):
        self.params = params

    def compute_kinematics_and_rom(self) -> Dict[str, Any]:
        """Calculates active ROM for each digit and joint."""
        xml_str = generate_hand_mjcf(self.params)
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        
        # Step at resting zero
        mujoco.mj_step(model, data)
        rest_angles = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): float(np.degrees(data.qpos[model.jnt_qposadr[j]]))
            for j in range(model.njnt)
        }
        
        # Actuate fully (all actuators tensioned)
        data.ctrl[:] = 200.0
        for _ in range(300):
            mujoco.mj_step(model, data)
            
        flexed_angles = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): float(np.degrees(data.qpos[model.jnt_qposadr[j]]))
            for j in range(model.njnt)
        }
        
        roms = {
            j_name: flexed_angles[j_name] - rest_angles[j_name]
            for j_name in rest_angles.keys()
        }
        
        return {
            "rest_angles_deg": rest_angles,
            "flexed_angles_deg": flexed_angles,
            "rom_deg": roms
        }

    def compute_force_transmission_efficiency(self) -> Dict[str, float]:
        """
        Calculates force transmission efficiency:
        eta = fingertip normal force / actuator tension force
        for each digit under isometric activation.
        """
        efficiencies = {}
        digits = ["index", "middle", "ring", "little"]
        
        for d_name in digits:
            pulley = self.params.pulleys[d_name]
            f_geom = self.params.fingers[d_name]
            # Moment arms at MCP, PIP, DIP
            r_mcp = pulley.mcp_radius
            r_pip = pulley.pip_radius
            r_dip = pulley.dip_radius
            
            # Phalanx lengths
            L_total = f_geom.proximal_len + f_geom.middle_len + f_geom.distal_len
            
            # In a flexor tendon chain: tau_total = F_tendon * (r_mcp + r_pip + r_dip)
            # Theoretical fingertip force ~ tau_total / L_effective
            # With internal flexure compliance and angular cosine losses:
            eta = (r_mcp + r_pip + r_dip) / L_total
            efficiencies[d_name] = float(np.clip(eta, 0.05, 0.65))
            
        # Thumb efficiency
        th_p = self.params.pulleys["thumb_flex"]
        th_geom = self.params.thumb
        th_L = th_geom.proximal_len + th_geom.distal_len
        efficiencies["thumb_flex"] = float(np.clip((th_p.mcp_radius + th_p.pip_radius) / th_L, 0.05, 0.65))
        efficiencies["thumb_opp"] = float(np.clip(self.params.pulleys["thumb_opp"].mcp_radius / th_geom.metacarpal_len, 0.05, 0.70))
        
        return efficiencies

    def compute_mass_and_volume(self) -> Dict[str, float]:
        """
        Estimates actuator mass, palm mass, forearm volume, total mass, and wrist torque.
        """
        hasel_vols = [act.volume_m3 for act in self.params.actuators.values()]
        hasel_masses = [act.mass_kg for act in self.params.actuators.values()]
        
        total_hasel_vol = sum(hasel_vols)
        total_hasel_mass = sum(hasel_masses)
        palm_mass = self.params.palm.mass
        
        # Estimate finger masses
        finger_mass_total = 0.085
        total_hand_mass = palm_mass + finger_mass_total + total_hasel_mass
        
        # Center of mass calculation (Z along forearm-palm axis)
        # Palm COM at z = palm.length / 2 = ~0.052m
        # Fingers COM at z = palm.length + 0.040m = ~0.145m
        # Forearm actuators COM at z = mean(actuator.position[2])
        act_z_mean = np.mean([act.position[2] for act in self.params.actuators.values()])
        
        com_z = (palm_mass * (self.params.palm.length * 0.5) +
                 finger_mass_total * (self.params.palm.length + 0.040) +
                 total_hasel_mass * act_z_mean) / total_hand_mass
                 
        # Wrist gravity torque at horizontal 90 deg extension (tau = m * g * r_com_relative_to_wrist)
        # Wrist interface is at z = 0.
        # Torque exerted on wrist by distal hand structures (palm + fingers):
        distal_mass = palm_mass + finger_mass_total
        distal_com_z = (palm_mass * (self.params.palm.length * 0.5) + finger_mass_total * (self.params.palm.length + 0.040)) / distal_mass
        wrist_gravity_torque = distal_mass * 9.81 * distal_com_z
        
        forearm_vol = self.params.wrist.forearm_cross_section_area * self.params.wrist.forearm_length
        
        return {
            "total_hasel_volume_cm3": float(total_hasel_vol * 1e6),
            "total_hasel_mass_g": float(total_hasel_mass * 1e3),
            "palm_mass_g": float(palm_mass * 1e3),
            "total_hand_mass_kg": float(total_hand_mass),
            "total_hand_mass_g": float(total_hand_mass * 1e3),
            "center_of_mass_z_mm": float(com_z * 1e3),
            "wrist_gravity_torque_Nm": float(wrist_gravity_torque),
            "forearm_volume_cm3": float(forearm_vol * 1e6)
        }
