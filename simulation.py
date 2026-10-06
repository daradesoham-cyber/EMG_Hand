"""
MuJoCo MJCF XML generator for the Parameterized Six-HASEL Prosthetic Hand.
Generates fully valid, dynamic MJCF XML models incorporating:
- Multi-body palm, 4 independent digits (MCP, PIP, DIP), and dual-axis thumb (CMC opposition, MP, IP)
- Passive compliance (flexure stiffness & damping on hinge joints)
- Accurate volar tendon routing through MCP, PIP, DIP and thumb pulleys
- Contact collision exclusion pairs to prevent artificial base joint penetrations
- Test grasp objects placed with initial physical clearance
"""

from typing import Dict, Any, Optional
import numpy as np
from parameters import HandParameters, load_default_parameters
from anthropometry import get_finger_base_positions, get_thumb_cmc_pose

def generate_hand_mjcf(params: HandParameters, object_type: Optional[str] = None, object_size: float = 0.035) -> str:
    """
    Constructs the complete MJCF XML string from hand parameters.
    
    Args:
        params: HandParameters dataclass instance
        object_type: Optional grasp test object ("cylinder", "sphere", "box", "thin", "small", "large")
        object_size: Dimension parameter for the object (radius/half-edge in meters)
    """
    palm = params.palm
    fingers = params.fingers
    thumb = params.thumb
    pulleys = params.pulleys
    
    bases = get_finger_base_positions(palm)
    thumb_cmc_pos, _ = get_thumb_cmc_pose(thumb)
    
    p_half_x = palm.width / 2.0
    p_half_y = palm.thickness / 2.0
    p_half_z = palm.length / 2.0
    
    xml_lines = [
        '<mujoco model="hasel_biomechatronic_hand">',
        '  <compiler angle="radian" coordinate="local" autolimits="true"/>',
        '  <option timestep="0.001" iterations="60" solver="Newton" gravity="0 0 0"/>',
        '  <default>',
        '    <joint limited="true" armature="0.002" damping="0.02" frictionloss="0.01"/>',
        '    <geom friction="1.8 0.01 0.0002" density="1100" solref="0.004 1" solimp="0.9 0.95 0.001"/>',
        '    <site size="0.003" rgba="1 0.2 0.2 0.8"/>',
        '  </default>',
        '  <contact>',
        '    <!-- Exclude adjacent proximal-to-palm contacts and adjacent base bones -->',
        '    <exclude body1="palm" body2="index_proximal"/>',
        '    <exclude body1="palm" body2="middle_proximal"/>',
        '    <exclude body1="palm" body2="ring_proximal"/>',
        '    <exclude body1="palm" body2="little_proximal"/>',
        '    <exclude body1="palm" body2="thumb_metacarpal"/>',
        '    <exclude body1="forearm" body2="thumb_metacarpal"/>',
        '    <exclude body1="index_proximal" body2="middle_proximal"/>',
        '    <exclude body1="middle_proximal" body2="ring_proximal"/>',
        '    <exclude body1="ring_proximal" body2="little_proximal"/>',
        '  </contact>',
        '  <asset>',
        '    <material name="palm_mat" rgba="0.25 0.28 0.32 1.0" specular="0.3" shininess="0.5"/>',
        '    <material name="finger_mat" rgba="0.75 0.78 0.82 1.0" specular="0.4" shininess="0.5"/>',
        '    <material name="pad_mat" rgba="0.88 0.45 0.38 1.0" specular="0.1" roughness="0.8"/>',
        '    <material name="object_mat" rgba="0.2 0.75 0.4 1.0" specular="0.5" shininess="0.6"/>',
        '    <material name="hasel_mat" rgba="0.1 0.4 0.8 0.7" specular="0.6"/>',
        '  </asset>',
        '  <worldbody>',
        '    <light directional="true" pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>',
        '',
        '    <!-- Forearm / Base chassis frame -->',
        '    <body name="forearm" pos="0 0 0">',
        '      <geom name="forearm_geom" type="box" size="0.045 0.035 0.100" pos="0 0 -0.100" rgba="0.3 0.3 0.35 0.4" contype="0" conaffinity="0"/>',
        '      <site name="wrist_anchor" pos="0 0 0"/>',
        ''
    ]
    
    # Add HASEL actuator housing visual markers / sites in forearm/palm
    for act_name, act in params.actuators.items():
        pos_str = f"{act.position[0]:.4f} {act.position[1]:.4f} {act.position[2]:.4f}"
        sz_w = act.width / 2.0
        sz_t = (act.thickness_per_layer * act.num_layers) / 2.0
        sz_l = act.resting_length / 2.0
        xml_lines.append(f'      <geom name="{act_name}_body" type="box" size="{sz_w:.4f} {sz_t:.4f} {sz_l:.4f}" pos="{pos_str}" material="hasel_mat" contype="0" conaffinity="0"/>')
        xml_lines.append(f'      <site name="{act_name}_site" pos="{pos_str}"/>')
        
    # Palm body
    xml_lines.extend([
        '',
        '      <!-- Palm Rigid Chassis -->',
        '      <body name="palm" pos="0 0 0">',
        f'        <geom name="palm_geom" type="box" size="{p_half_x:.4f} {p_half_y:.4f} {p_half_z:.4f}" pos="0 0 {p_half_z:.4f}" material="palm_mat" mass="{palm.mass:.3f}"/>',
        f'        <geom name="palm_pad" type="capsule" size="{p_half_y*0.7:.4f}" fromto="{-p_half_x*0.7:.4f} {-p_half_y:.4f} {p_half_z:.4f} {p_half_x*0.7:.4f} {-p_half_y:.4f} {p_half_z:.4f}" material="pad_mat"/>'
    ])
    
    # Palm volar tendon exits for the 4 fingers and thumb
    for f_name in ["index", "middle", "ring", "little"]:
        f_base = bases[f_name]
        f_pulley = pulleys[f_name]
        exit_z = f_base[2] - 0.010
        exit_y = -f_pulley.mcp_radius
        xml_lines.append(f'        <site name="{f_name}_palm_exit" pos="{f_base[0]:.4f} {exit_y:.4f} {exit_z:.4f}"/>')
        
    # Thumb tendon exits on palm
    xml_lines.append(f'        <site name="thumb_opp_exit" pos="0.005 -0.012 0.020"/>')
    xml_lines.append(f'        <site name="thumb_flex_exit" pos="{thumb_cmc_pos[0]:.4f} -0.015 {thumb_cmc_pos[2]-0.010:.4f}"/>')

    # 4 Fingers (Index, Middle, Ring, Little)
    for f_name in ["index", "middle", "ring", "little"]:
        f_geom = fingers[f_name]
        f_base = bases[f_name]
        f_pulley = pulleys[f_name]
        
        r_prox = f_geom.width_proximal / 2.0
        r_mid = (f_geom.width_proximal + f_geom.width_distal) / 4.0
        r_dist = f_geom.width_distal / 2.0
        
        # MCP Joint & Proximal Phalanx
        xml_lines.extend([
            f'',
            f'        <!-- === {f_name.upper()} FINGER === -->',
            f'        <body name="{f_name}_proximal" pos="{f_base[0]:.4f} {f_base[1]:.4f} {f_base[2]:.4f}">',
            f'          <joint name="{f_name}_mcp" type="hinge" axis="1 0 0" range="{f_geom.mcp_range[0]:.4f} {f_geom.mcp_range[1]:.4f}" stiffness="{f_geom.mcp_stiffness:.4f}" damping="{f_geom.mcp_damping:.4f}"/>',
            f'          <geom name="{f_name}_prox_geom" type="capsule" size="{r_prox:.4f}" fromto="0 0 0 0 0 {f_geom.proximal_len:.4f}" material="finger_mat"/>',
            f'          <site name="{f_name}_prox_mid" pos="0 {-f_pulley.mcp_radius:.4f} {f_geom.proximal_len*0.5:.4f}"/>',
            '',
            f'          <!-- PIP Joint & Middle Phalanx -->',
            f'          <body name="{f_name}_middle" pos="0 0 {f_geom.proximal_len:.4f}">',
            f'            <joint name="{f_name}_pip" type="hinge" axis="1 0 0" range="{f_geom.pip_range[0]:.4f} {f_geom.pip_range[1]:.4f}" stiffness="{f_geom.pip_stiffness:.4f}" damping="{f_geom.pip_damping:.4f}"/>',
            f'            <geom name="{f_name}_mid_geom" type="capsule" size="{r_mid:.4f}" fromto="0 0 0 0 0 {f_geom.middle_len:.4f}" material="finger_mat"/>',
            f'            <site name="{f_name}_mid_mid" pos="0 {-f_pulley.pip_radius:.4f} {f_geom.middle_len*0.5:.4f}"/>',
            '',
            f'            <!-- DIP Joint & Distal Phalanx + Fingertip -->',
            f'            <body name="{f_name}_distal" pos="0 0 {f_geom.middle_len:.4f}">',
            f'              <joint name="{f_name}_dip" type="hinge" axis="1 0 0" range="{f_geom.dip_range[0]:.4f} {f_geom.dip_range[1]:.4f}" stiffness="{f_geom.dip_stiffness:.4f}" damping="{f_geom.dip_damping:.4f}"/>',
            f'              <geom name="{f_name}_dist_geom" type="capsule" size="{r_dist:.4f}" fromto="0 0 0 0 0 {f_geom.distal_len:.4f}" material="finger_mat"/>',
            f'              <geom name="{f_name}_tip_pad" type="sphere" size="{r_dist*1.1:.4f}" pos="0 {-r_dist*0.3:.4f} {f_geom.distal_len:.4f}" material="pad_mat"/>',
            f'              <site name="{f_name}_dist_anchor" pos="0 {-f_pulley.dip_radius:.4f} {f_geom.distal_len*0.6:.4f}"/>',
            f'              <site name="{f_name}_fingertip" pos="0 0 {f_geom.distal_len:.4f}" size="0.002" rgba="0 1 0 1"/>',
            f'            </body>',
            f'          </body>',
            f'        </body>'
        ])
        
    # Thumb Subsystem (Dual-degree-of-freedom: CMC Opposition + MP/IP Flexion)
    r_th = thumb.width / 2.0
    th_flx_p = pulleys["thumb_flex"]
    xml_lines.extend([
        f'',
        f'        <!-- === THUMB SUBSYSTEM === -->',
        f'        <body name="thumb_metacarpal" pos="{thumb_cmc_pos[0]:.4f} {thumb_cmc_pos[1]:.4f} {thumb_cmc_pos[2]:.4f}">',
        f'          <!-- CMC Opposition Joint (Oblique Axis inward rotation) -->',
        f'          <joint name="thumb_cmc_opp" type="hinge" axis="-0.25 0.35 -0.9" range="{thumb.cmc_range[0]:.4f} {thumb.cmc_range[1]:.4f}" stiffness="{thumb.cmc_stiffness:.4f}" damping="{thumb.cmc_damping:.4f}"/>',
        f'          <geom name="thumb_meta_geom" type="capsule" size="{r_th:.4f}" fromto="0 0 0 0 -0.030 0.015" material="finger_mat"/>',
        f'          <site name="thumb_opp_anchor" pos="0 -0.025 0.012"/>',
        f'          <site name="thumb_meta_volar" pos="0 -0.010 0.005"/>',
        '',
        f'          <!-- Thumb Proximal & MP Flexion Joint -->',
        f'          <body name="thumb_proximal" pos="0 -0.030 0.015">',
        f'            <joint name="thumb_mp_flex" type="hinge" axis="1 0 0" range="{thumb.flex_range[0]:.4f} {thumb.flex_range[1]:.4f}" stiffness="{thumb.mp_stiffness:.4f}" damping="{thumb.mp_damping:.4f}"/>',
        f'            <geom name="thumb_prox_geom" type="capsule" size="{r_th*0.9:.4f}" fromto="0 0 0 0 0 {thumb.proximal_len:.4f}" material="finger_mat"/>',
        f'            <site name="thumb_prox_mid" pos="0 {-th_flx_p.mcp_radius:.4f} {thumb.proximal_len*0.5:.4f}"/>',
        '',
        f'            <!-- Thumb Distal & IP Flexion Joint -->',
        f'            <body name="thumb_distal" pos="0 0 {thumb.proximal_len:.4f}">',
        f'              <joint name="thumb_ip_flex" type="hinge" axis="1 0 0" range="{thumb.flex_range[0]*0.8:.4f} {thumb.flex_range[1]*0.8:.4f}" stiffness="{thumb.ip_stiffness:.4f}" damping="{thumb.ip_damping:.4f}"/>',
        f'              <geom name="thumb_dist_geom" type="capsule" size="{r_th*0.8:.4f}" fromto="0 0 0 0 0 {thumb.distal_len:.4f}" material="finger_mat"/>',
        f'              <geom name="thumb_tip_pad" type="sphere" size="{r_th*0.95:.4f}" pos="0 {-r_th*0.3:.4f} {thumb.distal_len:.4f}" material="pad_mat"/>',
        f'              <site name="thumb_dist_anchor" pos="0 {-th_flx_p.pip_radius:.4f} {thumb.distal_len*0.6:.4f}"/>',
        f'              <site name="thumb_tip" pos="0 0 {thumb.distal_len:.4f}" size="0.002" rgba="0 1 0 1"/>',
        f'            </body>',
        f'          </body>',
        f'        </body>',
        f'      </body> <!-- /palm -->',
        f'    </body> <!-- /forearm -->'
    ])
    
    # Grasp Objects presented with initial clearance in natural grasp envelope
    if object_type:
        if "pinch" in object_type or object_type == "small":
            # Precision pinch zone between index and thumb tips
            obj_x = 0.018
            obj_y = -0.045
            obj_z = 0.125
        else:
            # Power/cylindrical palmar grasp zone
            obj_x = 0.005
            obj_y = -0.048
            obj_z = 0.125
            
        xml_lines.extend([
            f'',
            f'    <!-- Grasp Test Object: {object_type} -->',
            f'    <body name="grasp_object" pos="{obj_x:.4f} {obj_y:.4f} {obj_z:.4f}">',
            f'      <!-- Statically presented test object for isometric grasp force & contact mapping -->'
        ])
        if object_type == "cylinder":
            xml_lines.append(f'      <geom name="target_geom" type="cylinder" size="{object_size:.4f} {object_size*1.8:.4f}" mass="0.120" material="object_mat"/>')
        elif object_type == "sphere":
            xml_lines.append(f'      <geom name="target_geom" type="sphere" size="{object_size:.4f}" mass="0.100" material="object_mat"/>')
        elif object_type == "box" or object_type == "cube":
            xml_lines.append(f'      <geom name="target_geom" type="box" size="{object_size:.4f} {object_size:.4f} {object_size:.4f}" mass="0.110" material="object_mat"/>')
        elif object_type == "thin":
            xml_lines.append(f'      <geom name="target_geom" type="box" size="0.025 0.005 0.030" mass="0.040" material="object_mat"/>')
        elif object_type == "small":
            xml_lines.append(f'      <geom name="target_geom" type="sphere" size="0.012" mass="0.020" material="object_mat"/>')
        elif object_type == "large":
            xml_lines.append(f'      <geom name="target_geom" type="cylinder" size="0.036 0.055" mass="0.180" material="object_mat"/>')
        else:
            xml_lines.append(f'      <geom name="target_geom" type="cylinder" size="0.025 0.045" mass="0.110" material="object_mat"/>')
        xml_lines.append(f'    </body>')

    xml_lines.extend([
        '  </worldbody>',
        '',
        '  <!-- Primary Tendon Transmission System (6 Independent Channels) -->',
        '  <tendon>'
    ])
    
    # 4 Digit flexor tendons
    for f_name in ["index", "middle", "ring", "little"]:
        xml_lines.extend([
            f'    <spatial name="{f_name}_flexor" width="0.0015" rgba="0.9 0.6 0.1 1">',
            f'      <site site="{f_name}_hasel_site"/>',
            f'      <site site="{f_name}_palm_exit"/>',
            f'      <site site="{f_name}_prox_mid"/>',
            f'      <site site="{f_name}_mid_mid"/>',
            f'      <site site="{f_name}_dist_anchor"/>',
            f'    </spatial>'
        ])
        
    # Thumb flexion tendon
    xml_lines.extend([
        f'    <spatial name="thumb_flexor" width="0.0015" rgba="0.9 0.6 0.1 1">',
        f'      <site site="thumb_flex_hasel_site"/>',
        f'      <site site="thumb_flex_exit"/>',
        f'      <site site="thumb_meta_volar"/>',
        f'      <site site="thumb_prox_mid"/>',
        f'      <site site="thumb_dist_anchor"/>',
        f'    </spatial>'
    ])
    
    # Thumb opposition tendon
    xml_lines.extend([
        f'    <spatial name="thumb_opposition" width="0.0015" rgba="0.1 0.7 0.9 1">',
        f'      <site site="thumb_opp_hasel_site"/>',
        f'      <site site="thumb_opp_exit"/>',
        f'      <site site="thumb_opp_anchor"/>',
        f'    </spatial>'
    ])
    
    xml_lines.extend([
        '  </tendon>',
        '',
        '  <!-- Actuators Driving Tendons (gear="-1" for tensile pulling) -->',
        '  <actuator>'
    ])
    
    for f_name in ["index", "middle", "ring", "little"]:
        f_max = params.actuators[f"{f_name}_hasel"].fmax
        xml_lines.append(f'    <motor name="{f_name}_actuator" tendon="{f_name}_flexor" ctrlrange="0 {f_max:.1f}" gear="-1"/>')
        
    th_f_max = params.actuators["thumb_flex_hasel"].fmax
    th_o_max = params.actuators["thumb_opp_hasel"].fmax
    xml_lines.append(f'    <motor name="thumb_flex_actuator" tendon="thumb_flexor" ctrlrange="0 {th_f_max:.1f}" gear="-1"/>')
    xml_lines.append(f'    <motor name="thumb_opp_actuator" tendon="thumb_opposition" ctrlrange="0 {th_o_max:.1f}" gear="-1"/>')
    
    xml_lines.extend([
        '  </actuator>',
        '</mujoco>'
    ])
    
    return '\n'.join(xml_lines)

def write_hand_xml(filepath: str, params: HandParameters, object_type: Optional[str] = None):
    xml_content = generate_hand_mjcf(params, object_type)
    with open(filepath, 'w') as f:
        f.write(xml_content)
