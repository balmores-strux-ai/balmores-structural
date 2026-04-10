"""
Simply supported beam: L = 6 m, w = 10 kN/m (uniform, full span).
Units: m, kN, kN·m. Uses Pynite from this repo.

Run:
  python Examples/Simple_Beam_6m_10kNm_diagrams.py

Outputs:
  - Console: support reactions + theory check
  - PNG diagrams next to this script (for viewing in Cursor)
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib import pyplot as plt

from Pynite import FEModel3D

L = 6.0
W = 10.0  # kN/m
E_KNM2 = 200e6  # ~200 GPa as kN/m^2
# Section 250 mm x 500 mm (strong axis vertical for Mz)
B, H = 0.25, 0.50
A = B * H
IY = H * B**3 / 12.0
IZ = B * H**3 / 12.0
J = (B * H**3 + H * B**3) / 12.0

model = FEModel3D()
model.add_node("N1", 0.0, 0.0, 0.0)
model.add_node("N2", L, 0.0, 0.0)
model.add_material("Steel", E_KNM2, 77e6, 0.3, 77.0)
model.add_section("S1", A, IY, IZ, J)
model.add_member("M1", "N1", "N2", "Steel", "S1")
# Same pin/roller pattern as Examples/Simple Beam - Uniform Load.py
model.def_support("N1", True, True, True, False, False, False)
model.def_support("N2", True, True, True, True, False, False)
model.add_member_dist_load("M1", "Fy", -W, -W, case="Case 1")

model.analyze_linear(check_statics=True)
combo = "Combo 1"
mem = model.members["M1"]

n = 200
xs = np.linspace(0.0, L, n)
V = np.array([mem.shear("Fy", float(x), combo) for x in xs])
Mz = np.array([mem.moment("Mz", float(x), combo) for x in xs])
d_y = np.array([mem.deflection("dy", float(x), combo) for x in xs])

# Slope theta = dd/dx (local elastic curve); central differences
theta = np.zeros_like(d_y)
theta[1:-1] = (d_y[2:] - d_y[:-2]) / (xs[2:] - xs[:-2])
theta[0] = (d_y[1] - d_y[0]) / (xs[1] - xs[0])
theta[-1] = (d_y[-1] - d_y[-2]) / (xs[-1] - xs[-2])

# Theory
R_theory = W * L / 2.0
M_mid_theory = W * L**2 / 8.0
d_mid_theory = 5.0 * W * L**4 / (384.0 * E_KNM2 * IZ)

n1, n2 = model.nodes["N1"], model.nodes["N2"]
Ry1 = n1.RxnFY[combo]
Ry2 = n2.RxnFY[combo]

print("=== Simply supported beam (Pynite) ===")
print(f"Span L = {L} m, w = {W} kN/m, combo = {combo}")
print(f"Support reactions (vertical, global Y, local Fy path):")
print(f"  N1 Ry = {Ry1:.4f} kN   (theory wL/2 = {R_theory:.4f} kN)")
print(f"  N2 Ry = {Ry2:.4f} kN   (theory wL/2 = {R_theory:.4f} kN)")
im = n // 2
print(f"Mid-span |V| ~ {abs(V[im]):.4f} kN (theory 0)")
print(f"Mid-span Mz = {Mz[im]:.4f} kN·m (theory wL^2/8 = {M_mid_theory:.4f} kN·m)")
print(f"Mid-span defl dy = {d_y[im]*1000:.4f} mm (theory 5wL^4/(384EI) = {d_mid_theory*1000:.4f} mm)")

out_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(out_dir, "Simple_Beam_6m_10kNm_4diagrams.png")

fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
fig.suptitle(
    f"Simply supported beam: L={L:g} m, w={W:g} kN/m (Pynite, {combo})",
    fontsize=12,
)

axes[0].plot(xs, V, "b-", lw=1.5)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_ylabel("Shear V (Fy)\n[kN]")
axes[0].grid(True, alpha=0.3)
axes[0].set_title("Shear diagram")

axes[1].plot(xs, Mz, "darkgreen", lw=1.5)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_ylabel("Moment Mz\n[kN·m]")
axes[1].grid(True, alpha=0.3)
axes[1].set_title("Moment diagram")

axes[2].plot(xs, theta * 1000.0, "m-", lw=1.5)
axes[2].axhline(0, color="k", lw=0.8)
axes[2].set_ylabel("Slope d(dy)/dx\n[mrad]")
axes[2].grid(True, alpha=0.3)
axes[2].set_title("Slope of elastic curve (numeric from deflection)")

axes[3].plot(xs, d_y * 1000.0, "r-", lw=1.5)
axes[3].axhline(0, color="k", lw=0.8)
axes[3].set_ylabel("Deflection dy\n[mm]")
axes[3].set_xlabel("Distance along member x [m]")
axes[3].grid(True, alpha=0.3)
axes[3].set_title("Deflection diagram (local dy)")

plt.tight_layout()
plt.savefig(png_path, dpi=150)
plt.close()
print(f"\nSaved diagram image (open in Cursor):\n  {png_path}")
