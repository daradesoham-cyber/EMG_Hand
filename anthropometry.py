"""
Anthropometry module for the Parameterized Prosthetic Hand.
Provides validated human anthropometric bounds, kinematic tree derivations,
segment coordinate calculations, and physical plausibility verification.
"""

from typing import Dict, Tuple, List
import numpy as np
from parameters import HandParameters, FingerSegmentGeometry, PalmGeometry, ThumbGeometry

def get_finger_base_positions(palm: PalmGeometry) -> Dict[str, np.ndarray]:
    """
    Computes MCP joint centers across the distal palm edge based on palm dimensions and spacing.
    Coordinates defined relative to palm frame (origin at carpal / wrist interface center).
    X: Lateral-Medial axis (+X towards thumb/radial side, -X towards ulnar)
    Y: Dorsal-Volar axis (+Y dorsal, -Y volar / grasp side)
    Z: Proximal-Distal axis (+Z distal towards fingertips)
    """
    half_width = palm.width / 2.0
    distal_z = palm.length
    
    # 4 finger bases arranged across transverse arch
    spacing = palm.finger_spacing
    # Index is radial (+X), Little is ulnar (-X)
    # Centers relative to palm midline:
    # Index: +1.5 * spacing, Middle: +0.5 * spacing, Ring: -0.5 * spacing, Little: -1.5 * spacing
    return {
        "index": np.array([1.5 * spacing, 0.0, distal_z]),
        "middle": np.array([0.5 * spacing, 0.0, distal_z + 0.003]), # slight natural middle prominence
        "ring": np.array([-0.5 * spacing, 0.0, distal_z]),
        "little": np.array([-1.5 * spacing, 0.0, distal_z - 0.004]) # slightly lower little finger base
    }

def get_thumb_cmc_pose(thumb: ThumbGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the thumb CMC joint location and orientation relative to palm frame.
    Thumb CMC sits at the antero-radial base of the palm.
    """
    pos = np.array(thumb.cmc_pos)
    euler = np.array(thumb.cmc_axis_euler)
    return pos, euler

def check_anthropometric_plausibility(params: HandParameters) -> Tuple[bool, List[str]]:
    """
    Validates that current parameters adhere to anthropometric feasibility constraints.
    Returns (is_valid, list_of_violations).
    """
    violations = []
    
    # Palm checks
    if not (0.075 <= params.palm.length <= 0.135):
        violations.append(f"Palm length {params.palm.length*1e3:.1f}mm out of anthropometric range [75, 135]mm")
    if not (0.065 <= params.palm.width <= 0.110):
        violations.append(f"Palm width {params.palm.width*1e3:.1f}mm out of range [65, 110]mm")
    if not (0.018 <= params.palm.thickness <= 0.040):
        violations.append(f"Palm thickness {params.palm.thickness*1e3:.1f}mm out of range [18, 40]mm")
        
    # Check that finger bases fit across palm width
    span = 3.0 * params.palm.finger_spacing + 0.018 # outer width of index & little MCPs
    if span > params.palm.width:
        violations.append(f"Finger span {span*1e3:.1f}mm exceeds palm width {params.palm.width*1e3:.1f}mm")

    # Segment length ratios (approximate human phalangeal proportions: proximal > middle > distal)
    for name, f in params.fingers.items():
        if f.proximal_len <= f.middle_len:
            violations.append(f"Finger {name}: proximal length ({f.proximal_len*1e3:.1f}mm) must exceed middle ({f.middle_len*1e3:.1f}mm)")
        if f.middle_len <= f.distal_len:
            violations.append(f"Finger {name}: middle length ({f.middle_len*1e3:.1f}mm) must exceed distal ({f.distal_len*1e3:.1f}mm)")
            
    return (len(violations) == 0), violations
