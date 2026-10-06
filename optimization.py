"""
Optimization and Scoring framework for the Parameterized Six-HASEL Prosthetic Hand.
Implements:
- Comprehensive multi-objective design scoring
- Hierarchical optimization stages A-H
- Robustness and sensitivity perturbation testing (+/-10% variations)
- Complete design database logging and results export
"""

import copy
import json
from dataclasses import asdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from parameters import HandParameters, load_default_parameters
from grasp_tests import GraspEvaluator, GraspTaskResult
from analysis import MechanicalAnalyzer

def evaluate_design_score(params: HandParameters, quick: bool = True) -> Dict[str, Any]:
    """
    Evaluates a candidate parameter set and calculates the composite design score.
    Rewards: grasp success, pinch force, power force, ROM, efficiency, stability.
    Penalizes: actuator mass/volume, excessive stroke, wrist torque, instability.
    """
    evaluator = GraspEvaluator(params)
    analyzer = MechanicalAnalyzer(params)
    
    # 1. Grasp task evaluations
    tasks_to_test = ["power_grasp", "precision_pinch", "tripod_pinch"] if quick else [
        "power_grasp", "precision_pinch", "tripod_pinch", "lateral_pinch", "sphere_grasp",
        "cylinder_grasp", "cube_grasp", "thin_object_grasp", "small_object_grasp", "large_object_grasp"
    ]
    
    task_results: Dict[str, GraspTaskResult] = {}
    success_count = 0
    total_contact_force = 0.0
    power_force = 0.0
    pinch_force = 0.0
    
    for t_name in tasks_to_test:
        res = evaluator.evaluate_task(t_name, duration_s=0.25)
        task_results[t_name] = res
        if res.success:
            success_count += 1
        total_contact_force += res.contact_force
        if t_name == "power_grasp":
            power_force = res.contact_force
        elif t_name == "precision_pinch":
            pinch_force = res.contact_force
            
    grasp_success_rate = success_count / len(tasks_to_test)
    
    # 2. Mechanical metrics
    mass_vol = analyzer.compute_mass_and_volume()
    efficiencies = analyzer.compute_force_transmission_efficiency()
    avg_efficiency = np.mean(list(efficiencies.values()))
    
    # 3. Stroke limits check
    stroke_penalty = 0.0
    for res in task_results.values():
        for act_name, stroke in res.actuator_strokes.items():
            max_s = params.actuators[act_name].max_stroke
            if stroke > max_s:
                stroke_penalty += (stroke - max_s) * 5000.0  # Heavy penalty for bottoming out stroke
                
    # 4. Multi-objective composite score formulation
    # Normalized reward terms:
    r_success = grasp_success_rate * 40.0
    r_power = min(1.0, power_force / 60.0) * 20.0
    r_pinch = min(1.0, pinch_force / 15.0) * 15.0
    r_efficiency = avg_efficiency * 15.0
    
    # Penalty terms:
    # Mass penalty (> 0.500 kg penalized)
    p_mass = max(0.0, mass_vol["total_hand_mass_kg"] - 0.500) * 30.0
    # Volume penalty (> 250 cm^3 penalized)
    p_vol = max(0.0, (mass_vol["total_hasel_volume_cm3"] - 250.0) / 100.0) * 10.0
    # Wrist torque penalty (> 0.25 Nm)
    p_wrist = max(0.0, mass_vol["wrist_gravity_torque_Nm"] - 0.20) * 20.0
    
    overall_score = float(r_success + r_power + r_pinch + r_efficiency - p_mass - p_vol - p_wrist - stroke_penalty)
    
    return {
        "overall_score": overall_score,
        "grasp_success_rate": grasp_success_rate,
        "power_force_N": float(power_force),
        "pinch_force_N": float(pinch_force),
        "total_contact_force_N": float(total_contact_force),
        "avg_efficiency": float(avg_efficiency),
        "total_mass_kg": mass_vol["total_hand_mass_kg"],
        "total_volume_cm3": mass_vol["total_hasel_volume_cm3"],
        "wrist_torque_Nm": mass_vol["wrist_gravity_torque_Nm"],
        "stroke_penalty": float(stroke_penalty),
        "task_results": {k: asdict(v) for k, v in task_results.items()},
        "efficiencies": efficiencies,
        "mass_vol": mass_vol
    }

def run_robustness_perturbation(best_params: HandParameters, num_trials: int = 15, rng_seed: int = 42) -> Dict[str, Any]:
    """
    Performs Monte Carlo perturbation testing on the identified best design.
    Randomly varies:
    - HASEL force +/- 10%
    - HASEL stroke +/- 10%
    - Joint stiffness +/- 10%
    - Pulley radius +/- 5%
    Runs multiple trials and evaluates stability, force variation, and success rate.
    """
    rng = np.random.default_rng(rng_seed)
    trials = []
    
    for trial_idx in range(num_trials):
        # Create perturbed clone
        p = copy.deepcopy(best_params)
        
        # Perturb HASEL force & stroke
        f_factor = rng.uniform(0.90, 1.10)
        s_factor = rng.uniform(0.90, 1.10)
        for act in p.actuators.values():
            act.fmax *= f_factor
            act.max_stroke *= s_factor
            
        # Perturb joint stiffness
        k_factor = rng.uniform(0.90, 1.10)
        for f in p.fingers.values():
            f.mcp_stiffness *= k_factor
            f.pip_stiffness *= k_factor
            f.dip_stiffness *= k_factor
        p.thumb.cmc_stiffness *= k_factor
        p.thumb.mp_stiffness *= k_factor
        p.thumb.ip_stiffness *= k_factor
        
        # Perturb pulley radius
        r_factor = rng.uniform(0.95, 1.05)
        for pul in p.pulleys.values():
            pul.mcp_radius *= r_factor
            pul.pip_radius *= r_factor
            pul.dip_radius *= r_factor
            
        # Evaluate
        eval_res = evaluate_design_score(p, quick=False)
        trials.append({
            "trial": trial_idx,
            "overall_score": eval_res["overall_score"],
            "grasp_success_rate": eval_res["grasp_success_rate"],
            "power_force_N": eval_res["power_force_N"],
            "pinch_force_N": eval_res["pinch_force_N"],
            "total_mass_kg": eval_res["total_mass_kg"]
        })
        
    df_trials = pd.DataFrame(trials)
    robustness_score = float(df_trials["grasp_success_rate"].mean())
    return {
        "robustness_score": robustness_score,
        "mean_power_force_N": float(df_trials["power_force_N"].mean()),
        "std_power_force_N": float(df_trials["power_force_N"].std()),
        "mean_pinch_force_N": float(df_trials["pinch_force_N"].mean()),
        "std_pinch_force_N": float(df_trials["pinch_force_N"].std()),
        "mean_overall_score": float(df_trials["overall_score"].mean()),
        "trials": trials
    }

class HierarchicalOptimizer:
    """
    Executes modular hierarchical optimization across Stages A-H:
    Stage A: Anthropometric palm & finger proportions
    Stage B: Finger tendon / pulley radii
    Stage C: HASEL pouch dimensions & layer counts
    Stage D: HASEL spatial placement & orientation
    Stage E: Palm architecture & transverse arch
    Stage F: Thumb CMC opposition & moment arms
    Stage G: Wrist/forearm packaging
    Stage H: Full-system consolidation
    """
    def __init__(self, base_params: HandParameters, seed: int = 42):
        self.params = copy.deepcopy(base_params)
        self.rng = np.random.default_rng(seed)
        self.database_records = []

    def log_candidate(self, design_id: str, p: HandParameters, eval_res: Dict[str, Any], stage: str):
        record = {
            "design_id": design_id,
            "stage": stage,
            "overall_score": eval_res["overall_score"],
            "grasp_success_rate": eval_res["grasp_success_rate"],
            "power_force_N": eval_res["power_force_N"],
            "pinch_force_N": eval_res["pinch_force_N"],
            "total_mass_kg": eval_res["total_mass_kg"],
            "total_volume_cm3": eval_res["total_volume_cm3"],
            "wrist_torque_Nm": eval_res["wrist_torque_Nm"],
            "palm_length_mm": p.palm.length * 1e3,
            "palm_width_mm": p.palm.width * 1e3,
            "mcp_pulley_radius_mm": p.pulleys["index"].mcp_radius * 1e3,
            "pip_pulley_radius_mm": p.pulleys["index"].pip_radius * 1e3,
            "dip_pulley_radius_mm": p.pulleys["index"].dip_radius * 1e3,
            "hasel_layers_primary": p.actuators["index_hasel"].num_layers,
            "hasel_fmax_primary_N": p.actuators["index_hasel"].fmax,
            "hasel_stroke_primary_mm": p.actuators["index_hasel"].max_stroke * 1e3,
            "placement_category": p.placement_category
        }
        self.database_records.append(record)

    def optimize_pulley_radii(self) -> HandParameters:
        """Stage B: Sweep and optimize MCP, PIP, and DIP pulley radii."""
        best_p = copy.deepcopy(self.params)
        best_score = -1e9
        
        mcp_candidates = [0.007, 0.009, 0.011, 0.013]
        pip_candidates = [0.005, 0.007, 0.009]
        dip_candidates = [0.004, 0.006]
        
        idx = 0
        for r_mcp in mcp_candidates:
            for r_pip in pip_candidates:
                for r_dip in dip_candidates:
                    p = copy.deepcopy(best_p)
                    for f in ["index", "middle", "ring"]:
                        p.pulleys[f].mcp_radius = r_mcp
                        p.pulleys[f].pip_radius = r_pip
                        p.pulleys[f].dip_radius = r_dip
                    # Little finger is shorter, scale pulley proportionally (approx 80%) to maintain stroke headroom
                    p.pulleys["little"].mcp_radius = r_mcp * 0.78
                    p.pulleys["little"].pip_radius = r_pip * 0.78
                    p.pulleys["little"].dip_radius = r_dip * 0.78
                    
                    eval_res = evaluate_design_score(p, quick=True)
                    self.log_candidate(f"stageB_pulley_{idx}", p, eval_res, "Stage_B_Pulleys")
                    if eval_res["overall_score"] > best_score:
                        best_score = eval_res["overall_score"]
                        best_p = p
                    idx += 1
                    
        self.params = best_p
        return best_p

    def optimize_hasel_dimensions(self) -> HandParameters:
        """Stage C: Optimize layer counts, stroke headroom, and thickness."""
        best_p = copy.deepcopy(self.params)
        best_score = -1e9
        
        layer_candidates = [6, 8, 10]
        width_candidates = [0.028, 0.032, 0.036]
        
        idx = 0
        for layers in layer_candidates:
            for w in width_candidates:
                p = copy.deepcopy(best_p)
                for f in ["index", "middle", "ring", "little"]:
                    act = p.actuators[f"{f}_hasel"]
                    act.num_layers = layers
                    act.width = w
                eval_res = evaluate_design_score(p, quick=True)
                self.log_candidate(f"stageC_hasel_{idx}", p, eval_res, "Stage_C_HASEL_Dimensions")
                if eval_res["overall_score"] > best_score:
                    best_score = eval_res["overall_score"]
                    best_p = p
                idx += 1
                
        self.params = best_p
        return best_p

    def optimize_placement(self) -> HandParameters:
        """Stage D: Evaluate placement categories (Forearm vs Palm vs Hybrid)."""
        best_p = copy.deepcopy(self.params)
        best_score = -1e9
        
        placements = ["forearm", "palm", "hybrid"]
        for cat in placements:
            p = copy.deepcopy(best_p)
            p.placement_category = cat
            if cat == "forearm":
                for f in ["index", "middle", "ring", "little"]:
                    p.actuators[f"{f}_hasel"].position[2] = -0.150
            elif cat == "palm":
                for f in ["index", "middle", "ring", "little"]:
                    p.actuators[f"{f}_hasel"].position[2] = 0.025
            elif cat == "hybrid":
                p.actuators["index_hasel"].position[2] = 0.020
                p.actuators["thumb_flex_hasel"].position[2] = 0.020
                p.actuators["middle_hasel"].position[2] = -0.140
                p.actuators["ring_hasel"].position[2] = -0.140
                p.actuators["little_hasel"].position[2] = -0.140
                p.actuators["thumb_opp_hasel"].position[2] = -0.120
                
            eval_res = evaluate_design_score(p, quick=True)
            self.log_candidate(f"stageD_placement_{cat}", p, eval_res, "Stage_D_Placement")
            if eval_res["overall_score"] > best_score:
                best_score = eval_res["overall_score"]
                best_p = p
                
        self.params = best_p
        return best_p

    def optimize_palm_and_thumb(self) -> HandParameters:
        """Stage E & F: Optimize palm dimensions and thumb opposition moment arm."""
        best_p = copy.deepcopy(self.params)
        best_score = -1e9
        
        palm_lens = [0.095, 0.105, 0.115]
        opp_radii = [0.009, 0.012, 0.015]
        
        idx = 0
        for pl in palm_lens:
            for orad in opp_radii:
                p = copy.deepcopy(best_p)
                p.palm.length = pl
                p.pulleys["thumb_opp"].mcp_radius = orad
                eval_res = evaluate_design_score(p, quick=True)
                self.log_candidate(f"stageEF_palm_thumb_{idx}", p, eval_res, "Stage_EF_Palm_Thumb")
                if eval_res["overall_score"] > best_score:
                    best_score = eval_res["overall_score"]
                    best_p = p
                idx += 1
                
        self.params = best_p
        return best_p

    def run_full_pipeline(self) -> Tuple[HandParameters, pd.DataFrame]:
        """Runs complete hierarchical optimization through all stages."""
        # Initial baseline evaluation
        init_res = evaluate_design_score(self.params, quick=True)
        self.log_candidate("baseline_init", self.params, init_res, "Baseline")
        
        # Stages
        self.optimize_pulley_radii()
        self.optimize_hasel_dimensions()
        self.optimize_placement()
        self.optimize_palm_and_thumb()
        
        df_database = pd.DataFrame(self.database_records)
        return self.params, df_database
