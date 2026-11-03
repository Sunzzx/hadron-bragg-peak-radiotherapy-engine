"""High-Fidelity Terminal ANSI / ASCII Visualizer for Hadron Radiotherapy.

Renders:
1. Depth-Dose Curves (Monoenergetic Bragg Peak vs Conventional 6 MV Photon X-Ray)
2. Spread-Out Bragg Peak (SOBP) Pristine Component Superposition
3. 2D Transversal CT Slice Anatomical Dose Colormap with Isodose Lines
4. Cumulative Dose-Volume Histogram (DVH) Clinical Target Verification
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .dvh import DVHCurve


def render_depth_dose_ascii(
    depths: np.ndarray,
    curves: Dict[str, np.ndarray],
    width: int = 70,
    height: int = 16,
) -> str:
    """Renders multi-curve depth-dose profile in terminal ASCII."""
    max_val = max(float(np.max(c)) for c in curves.values()) if curves else 1.0
    max_val = max(max_val, 1e-6)
    min_z = float(depths[0])
    max_z = float(depths[-1])

    grid = [[" " for _ in range(width)] for _ in range(height)]
    markers = ["*", "#", "+", "o", "x"]

    legend = []
    for idx, (name, curve) in enumerate(curves.items()):
        marker = markers[idx % len(markers)]
        legend.append(f"{marker} {name}")

        for col in range(width):
            z_target = min_z + (col / (width - 1)) * (max_z - min_z)
            val = float(np.interp(z_target, depths, curve))
            row = int((val / max_val) * (height - 1))
            row = max(0, min(height - 1, row))
            grid[height - 1 - row][col] = marker

    lines = ["  ┌" + "─" * width + "┐"]
    for r in range(height):
        val_at_row = max_val * (1.0 - r / (height - 1))
        prefix = f"{val_at_row:5.1f} |"
        lines.append(prefix + "".join(grid[r]) + "|")
    lines.append("      └" + "─" * width + "┘")
    lines.append(f"       {min_z:4.1f} cm" + " " * (width - 20) + f"{max_z:4.1f} cm")
    lines.append("       Legend: " + "  |  ".join(legend))
    return chr(10).join(lines)


def render_ct_dose_slice_ascii(
    dose_2d: np.ndarray,
    ptv_mask_2d: Optional[np.ndarray] = None,
    oar_mask_2d: Optional[np.ndarray] = None,
    width: int = 60,
    height: int = 22,
) -> str:
    """Renders a 2D transversal slice showing dose heatmap overlaid on anatomy."""
    ny, nx = dose_2d.shape
    ys = np.linspace(0, ny - 1, height).astype(int)
    xs = np.linspace(0, nx - 1, width).astype(int)

    sampled_dose = dose_2d[np.ix_(ys, xs)]
    max_d = float(np.max(sampled_dose)) if np.max(sampled_dose) > 0 else 1.0

    shades = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@", "█"]

    lines = ["  ┌" + "─" * width + "┐"]
    for r in range(height):
        row_chars = []
        for c in range(width):
            orig_y = ys[r]
            orig_x = xs[c]

            is_ptv = ptv_mask_2d[orig_y, orig_x] if ptv_mask_2d is not None else False
            is_oar = oar_mask_2d[orig_y, orig_x] if oar_mask_2d is not None else False
            val = sampled_dose[r, c] / max_d

            shade_idx = min(len(shades) - 1, int(val * (len(shades) - 1)))
            char = shades[shade_idx]

            if is_ptv and val < 0.2:
                char = "P"
            elif is_oar and val < 0.1:
                char = "B"
            row_chars.append(char)
        lines.append(f"  │{"".join(row_chars)}│")
    lines.append("  └" + "─" * width + "┘")
    lines.append(f"  [Scale: Max Dose = {max_d:.1f} Gy | P: PTV Tumor | B: Brainstem OAR | Dose: ' '->'.'->'#'->'█']")
    return chr(10).join(lines)


def render_dvh_ascii(
    dvh_dict: Dict[str, DVHCurve],
    width: int = 70,
    height: int = 16,
) -> str:
    """Renders multi-structure cumulative Dose-Volume Histogram in ASCII."""
    max_dose = max(c.max_dose for c in dvh_dict.values()) if dvh_dict else 60.0
    max_dose = max(max_dose, 1.0)

    grid = [[" " for _ in range(width)] for _ in range(height)]
    markers = ["#", "*", "+", "o", "x"]

    legend = []
    for idx, (name, curve) in enumerate(dvh_dict.items()):
        marker = markers[idx % len(markers)]
        legend.append(f"{marker} {name} (D50={curve.d50:.1f}Gy, D98={curve.d98:.1f}Gy)")

        for col in range(width):
            d_target = (col / (width - 1)) * max_dose
            vol = curve.v_at_dose(d_target)
            row = int((vol / 100.0) * (height - 1))
            row = max(0, min(height - 1, row))
            grid[height - 1 - row][col] = marker

    lines = ["  ┌" + "─" * width + "┐"]
    for r in range(height):
        vol_pct = 100.0 * (1.0 - r / (height - 1))
        prefix = f"{vol_pct:4.0f}% |"
        lines.append(prefix + "".join(grid[r]) + "|")
    lines.append("       └" + "─" * width + "┘")
    lines.append(f"        0.0 Gy" + " " * (width - 20) + f"{max_dose:5.1f} Gy")
    lines.append("       Structures: " + " | ".join(legend))
    return chr(10).join(lines)
