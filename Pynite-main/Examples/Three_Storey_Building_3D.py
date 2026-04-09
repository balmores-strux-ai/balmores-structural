"""
3-storey 3D moment frame — gravity analysis (10 kPa floor load).

Grid (m):
  X bays: 6, 8, 10  → x = 0, 6, 14, 24
  Y bays: 5, 9, 12, 6 (4th bay = 6 m — only three spans were specified)
  Storey heights: 5, 6, 10 → z = 0, 5, 11, 21

Loads: 10 kN/m² floor pressure; 50/50 two-way slab split onto X- and Y-running beams.

Units: m, kN, kN·m (moments), kPa. Steel E = 200 GPa.
"""
from __future__ import annotations

from Pynite import FEModel3D

# --- Geometry ---
XS = [0.0, 6.0, 14.0, 24.0]
YS = [0.0, 5.0, 14.0, 26.0, 32.0]
ZS = [0.0, 5.0, 11.0, 21.0]
STORY_H = [5.0, 6.0, 10.0]

Q_FLOOR = 10.0  # kPa = kN/m²
TWO_WAY = 0.5  # fraction of slab load to each beam system

# Tributary half-widths (m) for beams along X (perpendicular = Y)
def tributary_y(j: int) -> float:
    ny = len(YS)
    if j == 0:
        return (YS[1] - YS[0]) / 2
    if j == ny - 1:
        return (YS[j] - YS[j - 1]) / 2
    return (YS[j] - YS[j - 1]) / 2 + (YS[j + 1] - YS[j]) / 2


def tributary_x(i: int) -> float:
    nx = len(XS)
    if i == 0:
        return (XS[1] - XS[0]) / 2
    if i == nx - 1:
        return (XS[i] - XS[i - 1]) / 2
    return (XS[i] - XS[i - 1]) / 2 + (XS[i + 1] - XS[i]) / 2


def w_x_beam(j: int) -> float:
    return TWO_WAY * Q_FLOOR * tributary_y(j)


def w_y_beam(i: int) -> float:
    return TWO_WAY * Q_FLOOR * tributary_x(i)


def node_name(i: int, j: int, k: int) -> str:
    return f"n_{i}_{j}_{k}"


def main() -> None:
    m = FEModel3D()

    # --- Material (steel) ---
    E = 200e6  # kN/m²  (~200 GPa)
    G = 77e6
    nu = 0.3
    rho = 77.0  # kN/m³, nominal
    m.add_material("Steel", E, G, nu, rho)

    # --- Sections (preliminary prismatic; major axis = Iz) ---
    # Beam: ~0.40 m × 0.75 m rectangle (strong axis vertical in global Z for typical orientation)
    b_b, h_b = 0.40, 0.75
    A_b = b_b * h_b
    Iy_b = h_b * b_b**3 / 12
    Iz_b = b_b * h_b**3 / 12
    J_b = (b_b * h_b**3 + h_b * b_b**3) / 12  # approximate St. Venant

    # Column: ~0.45 m × 0.45 m box (solid equivalent for stiffness)
    b_c = 0.45
    A_c = b_c**2
    Iy_c = b_c**4 / 12
    Iz_c = Iy_c
    J_c = 0.5 * Iy_c  # rough torsion

    m.add_section("Beam", A_b, Iy_b, Iz_b, J_b)
    m.add_section("Column", A_c, Iy_c, Iz_c, J_c)

    nx, ny, nz = len(XS), len(YS), len(ZS)

    # --- Nodes ---
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                m.add_node(node_name(i, j, k), XS[i], YS[j], ZS[k])

    # --- Base fixity ---
    for i in range(nx):
        for j in range(ny):
            m.def_support(node_name(i, j, 0), True, True, True, True, True, True)

    # --- Columns ---
    col_members: list[str] = []
    for k in range(nz - 1):
        for j in range(ny):
            for i in range(nx):
                name = f"c_{i}_{j}_{k}"
                m.add_member(
                    name,
                    node_name(i, j, k),
                    node_name(i, j, k + 1),
                    "Steel",
                    "Column",
                )
                col_members.append(name)

    # --- Beams (X direction) at each elevated floor ---
    beam_x_members: list[tuple[str, int, int, int]] = []
    for k in range(1, nz):
        for j in range(ny):
            for i in range(nx - 1):
                name = f"bx_{i}_{j}_{k}"
                m.add_member(
                    name,
                    node_name(i, j, k),
                    node_name(i + 1, j, k),
                    "Steel",
                    "Beam",
                )
                beam_x_members.append((name, i, j, k))

    # --- Beams (Y direction) ---
    beam_y_members: list[tuple[str, int, int, int]] = []
    for k in range(1, nz):
        for i in range(nx):
            for j in range(ny - 1):
                name = f"by_{i}_{j}_{k}"
                m.add_member(
                    name,
                    node_name(i, j, k),
                    node_name(i, j + 1, k),
                    "Steel",
                    "Beam",
                )
                beam_y_members.append((name, i, j, k))

    case = "DL"
    # Uniform downward pressure via global FZ (negative Z)
    for name, i, j, k in beam_x_members:
        w = w_x_beam(j)
        m.add_member_dist_load(name, "FZ", -w, -w, case=case)

    for name, i, j, k in beam_y_members:
        w = w_y_beam(i)
        m.add_member_dist_load(name, "FZ", -w, -w, case=case)

    m.add_load_combo("SERVICE", {case: 1.0})

    print("Analyzing 3D frame (may take a few seconds)...")
    m.analyze(check_statics=True)
    combo = "SERVICE"

    def mag(res):
        v = res[0] if isinstance(res, tuple) else res
        return abs(float(v))

    print("\n" + "=" * 72)
    print("ASSUMED SECTIONS (preliminary - verify with your design code)")
    print("=" * 72)
    print(
        f"  Beams: rectangular {b_b:.2f} m x {h_b:.2f} m  |  "
        f"A = {A_b:.4f} m^2,  Iy = {Iy_b:.5f} m^4,  Iz = {Iz_b:.5f} m^4"
    )
    print(
        f"  Columns: square {b_c:.2f} m x {b_c:.2f} m  |  "
        f"A = {A_c:.4f} m^2,  I = {Iy_c:.5f} m^4"
    )
    print(
        f"  Steel: E = {E/1e6:.0f}e6 kN/m^2 (200 GPa), "
        f"G = {G/1e6:.0f}e6 kN/m^2"
    )
    print(f"  Y bay 4 taken as 6 m (you gave three Y spans; total Y = {YS[-1]:.0f} m).")
    print(f"  Floor load: {Q_FLOOR} kPa on all floors; {TWO_WAY:.0%} two-way split to each beam system.")

    # --- Global equilibrium ---
    print("\n" + "=" * 72)
    print("MAX ENVELOPE - BEAMS ALONG +X (gravity)")
    print("=" * 72)
    print(f"{'Member':<14} {'Floor z(m)':<10} {'Span':<6} {'|M|max':<12} {'|V|max':<12} {'defl_m':<12}")
    print("-" * 72)
    max_d_beam = 0.0
    for name, i, j, k in sorted(beam_x_members, key=lambda t: (t[3], t[2], t[1])):
        mem = m.members[name]
        span = XS[i + 1] - XS[i]
        zlev = ZS[k]
        Mz = mag(mem.max_moment("Mz", combo))
        My = mag(mem.max_moment("My", combo))
        Mmax = max(My, Mz)
        Vy = mag(mem.max_shear("Fy", combo))
        Vz = mag(mem.max_shear("Fz", combo))
        Vmax = max(Vy, Vz)
        try:
            d_y = mag(mem.max_deflection("dy", combo))
            d_z = mag(mem.max_deflection("dz", combo))
            dloc = max(d_y, d_z)
        except Exception:
            dloc = 0.0
        max_d_beam = max(max_d_beam, dloc)
        print(
            f"{name:<14} {zlev:<10.1f} {span:<6.1f} {Mmax:<12.2f} {Vmax:<12.2f} {dloc:<12.4f}"
        )
    print("* Local dy/dz deflection (m); max of the two.")

    print("\n" + "=" * 72)
    print("MAX ENVELOPE - BEAMS ALONG +Y")
    print("=" * 72)
    print(f"{'Member':<14} {'Floor z(m)':<10} {'Span':<6} {'|M|max':<12} {'|V|max':<12} {'defl_m':<12}")
    print("-" * 72)
    for name, i, j, k in sorted(beam_y_members, key=lambda t: (t[3], t[1], t[2])):
        mem = m.members[name]
        span = YS[j + 1] - YS[j]
        zlev = ZS[k]
        Mz = mag(mem.max_moment("Mz", combo))
        My = mag(mem.max_moment("My", combo))
        Mmax = max(My, Mz)
        Vy = mag(mem.max_shear("Fy", combo))
        Vz = mag(mem.max_shear("Fz", combo))
        Vmax = max(Vy, Vz)
        try:
            d_y = mag(mem.max_deflection("dy", combo))
            d_z = mag(mem.max_deflection("dz", combo))
            dloc = max(d_y, d_z)
        except Exception:
            dloc = 0.0
        max_d_beam = max(max_d_beam, dloc)
        print(
            f"{name:<14} {zlev:<10.1f} {span:<6.1f} {Mmax:<12.2f} {Vmax:<12.2f} {dloc:<12.4f}"
        )

    print("\n" + "=" * 72)
    print("COLUMNS - max |P|, |My|, |Mz| per vertical stack (i,j)")
    print("=" * 72)
    print(f"{'Stack':<10} {'Seg k':<8} {'|P| kN':<12} {'|My|':<12} {'|Mz|':<12} {'|DZ| roof mm':<12}")
    print("-" * 72)
    for i in range(nx):
        for j in range(ny):
            Pmax = Mymax = Mzmax = 0.0
            top_dz_mm = 0.0
            for k in range(nz - 1):
                name = f"c_{i}_{j}_{k}"
                mem = m.members[name]
                Pmax = max(Pmax, mag(mem.max_axial(combo)))
                Mymax = max(Mymax, mag(mem.max_moment("My", combo)))
                Mzmax = max(Mzmax, mag(mem.max_moment("Mz", combo)))
            nd = m.nodes[node_name(i, j, nz - 1)]
            top_dz_mm = abs(nd.DZ[combo]) * 1000.0
            print(
                f"({i},{j}){'':<5} 0-{nz-2}{'':<4} {Pmax:<12.1f} {Mymax:<12.2f} {Mzmax:<12.2f} {top_dz_mm:<12.3f}"
            )

    # Drift: gravity-only → lateral displacements ~ 0
    print("\n" + "=" * 72)
    print("DRIFT & VERTICAL DISPLACEMENT")
    print("=" * 72)
    print(
        "  Lateral inter-storey drift (DX, DY): under symmetric gravity-only loading, "
        "these are essentially zero (machine noise)."
    )
    corner = node_name(nx - 1, ny - 1, nz - 1)
    n = m.nodes[corner]
    print(
        f"  Roof corner {corner}: DX={n.DX[combo]:.2e} m, DY={n.DY[combo]:.2e} m, "
        f"DZ={n.DZ[combo]*1000:.3f} mm (downward negative)."
    )
    for s, (z_lo, z_hi) in enumerate(
        [(ZS[0], ZS[1]), (ZS[1], ZS[2]), (ZS[2], ZS[3])], start=1
    ):
        n_lo = m.nodes[node_name(0, 0, s - 1)]
        n_hi = m.nodes[node_name(0, 0, s)]
        dz_rel = (n_hi.DZ[combo] - n_lo.DZ[combo]) * 1000.0
        h = STORY_H[s - 1]
        print(
            f"  Storey {s} (z {z_lo:.0f}-{z_hi:.0f} m): "
            f"relative vertical slip at grid (0,0) = {dz_rel:.3f} mm over h = {h:.1f} m"
        )

    print(f"\n  Largest reported beam local deflection magnitude: {max_d_beam*1000:.2f} mm")
    print("=" * 72)


if __name__ == "__main__":
    main()
