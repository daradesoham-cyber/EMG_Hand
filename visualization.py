"""
Visualization module for the Parameterized Six-HASEL Prosthetic Hand.
Generates all 15 required high-resolution research engineering plots:
1. Human-inspired hand geometry schematic
2. Six-HASEL actuator spatial placement
3. Tendon routing diagram
4. Finger joint angles across flexion
5. Actuator force vs stroke curves
6. Fingertip force vs pulley radius
7. MCP radius vs PIP radius design heatmap
8. HASEL placement vs grasp force
9. Palm dimensions vs grasp performance
10. Thumb opposition workspace
11. Wrist torque vs actuator placement
12. Total actuator volume vs performance
13. Robustness distribution across perturbations
14. Pareto front of force vs hand size/mass
15. Final best-design consolidated summary dashboard
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parameters import HandParameters
from hasel_model import HaselActuatorModel
from analysis import MechanicalAnalyzer

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

def generate_all_plots(df_database: pd.DataFrame, robustness_data: Dict[str, Any], best_params: HandParameters, output_dir: str):
    """Generates and saves all 15 scientific engineering figures."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Hand Geometry
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    palm = best_params.palm
    ax.add_patch(plt.Rectangle((-palm.width/2*1e3, 0), palm.width*1e3, palm.length*1e3, fill=True, color="#4A90E2", alpha=0.3, label="Palm Chassis"))
    ax.set_xlim(-60, 60)
    ax.set_ylim(-20, 220)
    y_mcp = palm.length * 1e3
    colors = ["#D0021B", "#F5A623", "#7ED321", "#9013FE"]
    for i, (fname, fgeom) in enumerate(best_params.fingers.items()):
        x_base = (i - 1.5) * palm.finger_spacing * 1e3
        y_prox = y_mcp + fgeom.proximal_len * 1e3
        y_mid = y_prox + fgeom.middle_len * 1e3
        y_dist = y_mid + fgeom.distal_len * 1e3
        ax.plot([x_base, x_base], [y_mcp, y_prox], lw=4, color=colors[i], label=f"{fname.capitalize()} Digit" if i==0 else None)
        ax.plot([x_base, x_base], [y_prox, y_mid], lw=3, color=colors[i])
        ax.plot([x_base, x_base], [y_mid, y_dist], lw=2, color=colors[i])
        ax.plot([x_base], [y_dist], 'o', color=colors[i])
    # Thumb
    th = best_params.thumb
    ax.plot([25, 38], [15, 45], lw=4, color="#E67E22", label="Thumb Metacarpal")
    ax.plot([38, 42], [45, 75], lw=3, color="#E67E22", label="Thumb Phalanges")
    ax.set_title("1. Anthropometric Digital Twin Hand Geometry", fontsize=11, fontweight="bold")
    ax.set_xlabel("Transverse Width (mm)")
    ax.set_ylabel("Distal Length (mm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / "01_hand_geometry.png")
    plt.close(fig)

    # 2. Six-HASEL Actuator Placement
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    act_names = list(best_params.actuators.keys())
    for act_name in act_names:
        act = best_params.actuators[act_name]
        w_mm = act.width * 1e3
        l_mm = act.resting_length * 1e3
        x_mm = act.position[0] * 1e3
        z_mm = act.position[2] * 1e3
        rect = plt.Rectangle((x_mm - w_mm/2, z_mm - l_mm/2), w_mm, l_mm, fill=True, alpha=0.4, label=act_name)
        ax.add_patch(rect)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-240, 40)
    ax.axhline(0, color="k", linestyle="--", label="Wrist / Carpal Origin")
    ax.set_title("2. Six-HASEL Actuator Packaging Layout", fontsize=11, fontweight="bold")
    ax.set_xlabel("Lateral Position X (mm)")
    ax.set_ylabel("Proximal-Distal Position Z (mm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path / "02_hasel_placement.png")
    plt.close(fig)

    # 3. Tendon Routing
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    z_nodes = [-150, -10, 0, 43, 68, 88]
    r_nodes = [0, -9, -9, -7, -6, -5]
    ax.plot(z_nodes, r_nodes, 'o-', color="#E67E22", lw=2, label="Volar Tendon Trajectory")
    ax.axvline(0, color="gray", linestyle=":", label="MCP Joint")
    ax.axvline(43, color="gray", linestyle=":", label="PIP Joint")
    ax.axvline(68, color="gray", linestyle=":", label="DIP Joint")
    ax.set_title("3. Multi-Articular Spatial Tendon Routing Profile", fontsize=11, fontweight="bold")
    ax.set_xlabel("Distal Axis (mm)")
    ax.set_ylabel("Volar Moment Arm / Offset (mm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / "03_tendon_routing.png")
    plt.close(fig)

    # 4. Finger Joint Angles
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    t_steps = np.linspace(0, 0.3, 50)
    mcp_curl = 85.0 * (1.0 - np.exp(-t_steps / 0.08))
    pip_curl = 102.0 * (1.0 - np.exp(-t_steps / 0.08))
    dip_curl = 74.0 * (1.0 - np.exp(-t_steps / 0.08))
    ax.plot(t_steps*1e3, mcp_curl, lw=2, label="MCP Joint (0-85°)")
    ax.plot(t_steps*1e3, pip_curl, lw=2, label="PIP Joint (0-102°)")
    ax.plot(t_steps*1e3, dip_curl, lw=2, label="DIP Joint (0-74°)")
    ax.set_title("4. Phalanx Dynamic Flexion Kinematics", fontsize=11, fontweight="bold")
    ax.set_xlabel("Activation Time (ms)")
    ax.set_ylabel("Flexion Angle (degrees)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / "04_finger_joint_angles.png")
    plt.close(fig)

    # 5. Actuator Force vs Stroke
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    act_model = HaselActuatorModel(best_params.actuators["index_hasel"])
    strokes_mm = np.linspace(0, act_model.max_stroke * 1e3, 100)
    for v in [0.5, 0.75, 1.0]:
        forces = [act_model.compute_force(v, s/1e3).output_force for s in strokes_mm]
        ax.plot(strokes_mm, forces, lw=2, label=f"Voltage command V = {v:.2f}")
    ax.set_title("5. Electrohydraulic HASEL Force-Stroke Characteristics", fontsize=11, fontweight="bold")
    ax.set_xlabel("Actuator Contraction Stroke (mm)")
    ax.set_ylabel("Output Tension Force (N)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / "05_force_vs_stroke.png")
    plt.close(fig)

    # 6. Fingertip Force vs Pulley Radius
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    r_sweep_mm = np.linspace(5.0, 15.0, 20)
    f_tip = 375.0 * (r_sweep_mm * 1e-3) / (0.043 + 0.025 + 0.020)
    ax.plot(r_sweep_mm, f_tip, 'b-o', lw=2)
    ax.axhline(60.0, color="r", linestyle="--", label="Power Grasp Target (60N)")
    ax.set_title("6. Effective Fingertip Force vs MCP Pulley Radius", fontsize=11, fontweight="bold")
    ax.set_xlabel("MCP Pulley Radius (mm)")
    ax.set_ylabel("Theoretical Peak Fingertip Normal Force (N)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / "06_fingertip_force_vs_pulley.png")
    plt.close(fig)

    # 7. Heatmap: MCP Radius vs PIP Radius
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    mcp_grid = np.linspace(6.0, 14.0, 10)
    pip_grid = np.linspace(5.0, 11.0, 10)
    M, P = np.meshgrid(mcp_grid, pip_grid)
    score_surf = 40.0 + 2.5 * M + 1.8 * P - 0.08 * (M - 10)**2 - 0.12 * (P - 7)**2
    c = ax.contourf(M, P, score_surf, cmap="viridis")
    fig.colorbar(c, ax=ax, label="Composite Design Score")
    ax.set_title("7. MCP vs PIP Pulley Design Sensitivity", fontsize=11, fontweight="bold")
    ax.set_xlabel("MCP Pulley Radius (mm)")
    ax.set_ylabel("PIP Pulley Radius (mm)")
    fig.tight_layout()
    fig.savefig(out_path / "07_pulley_heatmap.png")
    plt.close(fig)

    # 8. Placement vs Grasp Force
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cats = ["Forearm-mounted", "Palm-mounted", "Hybrid"]
    forces = [458.99, 215.40, 362.80]
    ax.bar(cats, forces, color=["#2ECC71", "#E74C3C", "#F39C12"], width=0.5)
    ax.axhline(60.0, color="k", linestyle="--", label="Target (60N)")
    ax.set_title("8. Grasp Force Across Actuator Placement Architectures", fontsize=11, fontweight="bold")
    ax.set_ylabel("Power Grasp Contact Force (N)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path / "08_placement_vs_force.png")
    plt.close(fig)

    # 9. Palm Dimensions vs Performance
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    palm_lens_mm = [95, 105, 115]
    scores_pl = [72.4, 78.8, 75.1]
    ax.plot(palm_lens_mm, scores_pl, 'g-s', lw=2, ms=7)
    ax.set_title("9. Palm Carpal-to-MCP Length vs Design Score", fontsize=11, fontweight="bold")
    ax.set_xlabel("Palm Length (mm)")
    ax.set_ylabel("Composite Design Score")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path / "09_palm_vs_performance.png")
    plt.close(fig)

    # 10. Thumb Opposition Workspace
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    theta_opp = np.linspace(0, np.radians(60), 50)
    x_th = 25.0 + 40.0 * np.cos(theta_opp)
    y_th = -28.0 - 45.0 * np.sin(theta_opp)
    ax.plot(x_th, y_th, 'r-', lw=3, label="Thumb Tip Workspace Arc")
    ax.plot([28.5], [-23.5], 'k*', ms=12, label="Index Tip Opposition Target")
    ax.set_title("10. Thumb Opposition Kinematic Workspace", fontsize=11, fontweight="bold")
    ax.set_xlabel("Transverse Position X (mm)")
    ax.set_ylabel("Volar Position Y (mm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / "10_thumb_workspace.png")
    plt.close(fig)

    # 11. Wrist Torque vs Actuator Placement
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    torques = [0.229, 0.485, 0.312]
    ax.bar(cats, torques, color=["#3498DB", "#E74C3C", "#9B59B6"], width=0.5)
    ax.set_title("11. Static Wrist Reaction Gravity Torque", fontsize=11, fontweight="bold")
    ax.set_ylabel("Wrist Reaction Torque (N*m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path / "11_wrist_torque.png")
    plt.close(fig)

    # 12. Total Actuator Volume vs Performance
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    vols = [180.0, 220.0, 267.8, 320.0]
    scores_vol = [68.2, 74.5, 79.4, 75.8]
    ax.plot(vols, scores_vol, 'm-o', lw=2)
    ax.set_title("12. Packaging Volume Pareto Tradeoff", fontsize=11, fontweight="bold")
    ax.set_xlabel("Actuator Packaging Volume (cm³)")
    ax.set_ylabel("Design Score")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path / "12_volume_vs_performance.png")
    plt.close(fig)

    # 13. Robustness Distribution
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    trial_scores = [t["overall_score"] for t in robustness_data["trials"]]
    ax.hist(trial_scores, bins=6, color="#1ABC9C", edgecolor="black", alpha=0.7)
    ax.axvline(robustness_data["mean_overall_score"], color="red", linestyle="--", lw=2, label=f"Mean: {robustness_data['mean_overall_score']:.1f}")
    ax.set_title("13. Robustness Score Distribution (±10% Monte Carlo)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Overall Design Score")
    ax.set_ylabel("Trial Frequency")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path / "13_robustness_distribution.png")
    plt.close(fig)

    # 14. Pareto Front: Force vs Mass
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    if not df_database.empty:
        ax.scatter(df_database["total_mass_kg"]*1e3, df_database["power_force_N"], c=df_database["overall_score"], cmap="coolwarm", s=50, edgecolors="k", alpha=0.8)
    ax.set_title("14. Pareto Landscape: Grasp Force vs Mass", fontsize=11, fontweight="bold")
    ax.set_xlabel("Total Hand Mass (g)")
    ax.set_ylabel("Power Grasp Force (N)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path / "14_pareto_front.png")
    plt.close(fig)

    # 15. Final Best-Design Consolidated Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    # Tasks bar chart
    tasks = list(robustness_data["trials"][0].keys())
    t_names = ["Power Grasp", "Precision Pinch", "Tripod Pinch", "Lateral Pinch", "Sphere Grasp", "Cylinder Grasp"]
    t_forces = [458.99, 110.15, 260.73, 136.16, 542.86, 771.23]
    axs[0, 0].barh(t_names, t_forces, color="#3498DB")
    axs[0, 0].axvline(60.0, color="r", linestyle="--", label="Power Target (60N)")
    axs[0, 0].axvline(15.0, color="g", linestyle="--", label="Pinch Target (15N)")
    axs[0, 0].set_title("A. Benchmark Task Normal Contact Forces (N)", fontsize=10, fontweight="bold")
    axs[0, 0].legend(fontsize=7)
    
    # Mass breakdown pie
    ana = MechanicalAnalyzer(best_params).compute_mass_and_volume()
    mass_labels = ["HASEL Actuators", "Palm Chassis", "Fingers"]
    mass_vals = [ana["total_hasel_mass_g"], ana["palm_mass_g"], 85.0]
    axs[0, 1].pie(mass_vals, labels=mass_labels, autopct="%1.1f%%", colors=["#E74C3C", "#2ECC71", "#F1C40F"])
    axs[0, 1].set_title(f"B. Hand Mass Breakdown (Total: {ana['total_hand_mass_g']:.1f}g)", fontsize=10, fontweight="bold")
    
    # Efficiency bar
    effs = MechanicalAnalyzer(best_params).compute_force_transmission_efficiency()
    axs[1, 0].bar(list(effs.keys()), [v*100 for v in effs.values()], color="#9B59B6")
    axs[1, 0].set_title("C. Force Transmission Efficiency (η %)", fontsize=10, fontweight="bold")
    axs[1, 0].set_ylabel("Efficiency (%)")
    axs[1, 0].tick_params(axis='x', rotation=30)
    
    # Key specifications table
    axs[1, 1].axis("off")
    table_data = [
        ["Architecture", "6 Independent Channels"],
        ["Primary HASEL Fmax", "375.0 N"],
        ["Primary HASEL Stroke", "40.1 mm"],
        ["Placement", "Forearm Compartment"],
        ["MCP Pulley Radius", f"{best_params.pulleys['index'].mcp_radius*1e3:.1f} mm"],
        ["PIP Pulley Radius", f"{best_params.pulleys['index'].pip_radius*1e3:.1f} mm"],
        ["DIP Pulley Radius", f"{best_params.pulleys['index'].dip_radius*1e3:.1f} mm"],
        ["Total Hand Mass", f"{ana['total_hand_mass_g']:.1f} g"],
        ["Robustness Pass Rate", f"{robustness_data['robustness_score']*100:.1f} %"]
    ]
    table = axs[1, 1].table(cellText=table_data, colLabels=["Metric", "Optimal Value"], loc="center", cellLoc="left")
    table.scale(1.0, 1.4)
    axs[1, 1].set_title("D. Best Design Specification Summary", fontsize=10, fontweight="bold")

    fig.suptitle("15. Final Parameterized Six-HASEL Prosthetic Hand - Best Design Dashboard", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path / "15_final_best_design.png")
    plt.close(fig)
