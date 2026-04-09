"""
Stress test: tall 3D moment frame using Pynite from this repo.
Reports base reactions, statics, and envelopes of internal forces.

Units: m, kN, kN*m. Run: python Examples/Highrise_Stress_Test.py
"""
from __future__ import annotations

import time

from Pynite import FEModel3D

# --- "Crazy" but still solvable on a typical PC ---
# Scale up N_STORY / grid in steps; very tall models take minutes and large RAM.
N_STORY = 100  # ~400 m height; increase/decrease to probe limits
STORY_H = 4.0  # m
BAY = 8.0  # m (square bays)
NX = 5  # column lines in X
NY = 5  # column lines in Y
Q_KPA = 8.0  # floor pressure kN/m^2
TWO_WAY = 0.5

XS = [BAY * i for i in range(NX)]
YS = [BAY * i for i in range(NY)]
NZ = N_STORY + 1
ZS = [STORY_H * k for k in range(NZ)]


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


def node_name(i: int, j: int, k: int) -> str:
    return f"n_{i}_{j}_{k}"


def main() -> None:
    t0 = time.perf_counter()
    m = FEModel3D()

    E = 200e6
    G = 77e6
    nu = 0.3
    m.add_material("Steel", E, G, nu, 77.0)

    # Stocky sections (stability / conditioning)
    bc, bb, hb = 1.0, 0.55, 0.95
    Ac = bc**2
    Ic = bc**4 / 12
    Ab = bb * hb
    Iy_b = hb * bb**3 / 12
    Iz_b = bb * hb**3 / 12
    Jb = (bb * hb**3 + hb * bb**3) / 12
    m.add_section("Col", Ac, Ic, Ic, 0.7 * Ic)
    m.add_section("Bm", Ab, Iy_b, Iz_b, Jb)

    nx, ny, nz = NX, NY, NZ

    print(
        f"Building model: {nx}x{ny} columns, {N_STORY} storeys x {STORY_H} m ...",
        flush=True,
    )
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                m.add_node(node_name(i, j, k), XS[i], YS[j], ZS[k])

    for i in range(nx):
        for j in range(ny):
            m.def_support(node_name(i, j, 0), True, True, True, True, True, True)

    for k in range(nz - 1):
        for j in range(ny):
            for i in range(nx):
                m.add_member(
                    f"c_{i}_{j}_{k}",
                    node_name(i, j, k),
                    node_name(i, j, k + 1),
                    "Steel",
                    "Col",
                )

    case = "DL"
    for k in range(1, nz):
        for j in range(ny):
            for i in range(nx - 1):
                name = f"bx_{i}_{j}_{k}"
                m.add_member(
                    name,
                    node_name(i, j, k),
                    node_name(i + 1, j, k),
                    "Steel",
                    "Bm",
                )
                w = TWO_WAY * Q_KPA * tributary_y(j)
                m.add_member_dist_load(name, "FZ", -w, -w, case=case)
        for i in range(nx):
            for j in range(ny - 1):
                name = f"by_{i}_{j}_{k}"
                m.add_member(
                    name,
                    node_name(i, j, k),
                    node_name(i, j + 1, k),
                    "Steel",
                    "Bm",
                )
                w = TWO_WAY * Q_KPA * tributary_x(i)
                m.add_member_dist_load(name, "FZ", -w, -w, case=case)

    m.add_load_combo("SERVICE", {case: 1.0})
    combo = "SERVICE"

    t1 = time.perf_counter()
    print(f"Geometry built in {t1 - t0:.2f} s. Running linear analysis ...", flush=True)
    # Statics check loops all members/nodes — very slow on huge models
    m.analyze_linear(check_statics=False)
    t2 = time.perf_counter()
    print(f"Analysis finished in {t2 - t1:.2f} s (total {t2 - t0:.2f} s).\n", flush=True)

    def mag(res):
        v = res[0] if isinstance(res, tuple) else res
        return abs(float(v))

    # --- Base reactions ---
    print("=" * 72)
    print("BASE SUPPORT REACTIONS (each column foot, combo SERVICE)")
    print("=" * 72)
    sum_fx = sum_fy = sum_fz = 0.0
    sum_mx = sum_my = sum_mz = 0.0
    corners = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
    for i in range(nx):
        for j in range(ny):
            nd = m.nodes[node_name(i, j, 0)]
            fx = nd.RxnFX[combo]
            fy = nd.RxnFY[combo]
            fz = nd.RxnFZ[combo]
            sum_fx += fx
            sum_fy += fy
            sum_fz += fz
            sum_mx += nd.RxnMX[combo]
            sum_my += nd.RxnMY[combo]
            sum_mz += nd.RxnMZ[combo]
            if (i, j) in corners:
                print(
                    f"  Corner ({i},{j}): RxnFX={fx:10.2f}  RxnFY={fy:10.2f}  "
                    f"RxnFZ={fz:12.2f} kN"
                )
                print(
                    f"            RxnMX={nd.RxnMX[combo]:10.2f}  RxnMY={nd.RxnMY[combo]:10.2f}  "
                    f"RxnMZ={nd.RxnMZ[combo]:10.2f} kN*m"
                )

    print(f"\n  SUM all base RxnFX = {sum_fx:.2f} kN")
    print(f"  SUM all base RxnFY = {sum_fy:.2f} kN")
    print(f"  SUM all base RxnFZ = {sum_fz:.2f} kN  (vertical equilibrium)")
    print(f"  SUM all base RxnMX = {sum_mx:.2f} kN*m")
    print(f"  SUM all base RxnMY = {sum_my:.2f} kN*m")
    print(f"  SUM all base RxnMZ = {sum_mz:.2f} kN*m")

    area = (XS[-1] - XS[0]) * (YS[-1] - YS[0])
    applied_vert = Q_KPA * area * N_STORY
    print(f"\n  Nominal total floor load (pressure x tributary footprint x floors): ~{applied_vert:.0f} kN")
    print(f"  (Two-way split; see statics table above for exact global balance.)")

    # --- Envelopes ---
    print("\n" + "=" * 72)
    print("INTERNAL FORCES — ENVELOPE OVER ALL MEMBERS (SERVICE)")
    print("=" * 72)
    max_p = max_my = max_mz = max_vy = max_vz = 0.0
    max_p_n = max_my_n = max_mz_n = ""
    for name, mem in m.members.items():
        p = mag(mem.max_axial(combo))
        my = mag(mem.max_moment("My", combo))
        mz = mag(mem.max_moment("Mz", combo))
        vy = mag(mem.max_shear("Fy", combo))
        vz = mag(mem.max_shear("Fz", combo))
        if p > max_p:
            max_p, max_p_n = p, name
        if my > max_my:
            max_my, max_my_n = my, name
        if mz > max_mz:
            max_mz, max_mz_n = mz, name
        max_vy = max(max_vy, vy)
        max_vz = max(max_vz, vz)

    print(f"  Max |axial| P   = {max_p:.2f} kN   (member {max_p_n})")
    print(f"  Max |My|       = {max_my:.2f} kN*m (member {max_my_n})")
    print(f"  Max |Mz|       = {max_mz:.2f} kN*m (member {max_mz_n})")
    print(f"  Max |Fy| shear = {max_vy:.2f} kN")
    print(f"  Max |Fz| shear = {max_vz:.2f} kN")

    # Sample: bottom-storey column and top-floor beam
    c_mid = f"c_{nx//2}_{ny//2}_0"
    mem_c = m.members[c_mid]
    print(f"\n  Sample bottom column {c_mid}:")
    print(f"    |P|={mag(mem_c.max_axial(combo)):.2f} kN, |My|={mag(mem_c.max_moment('My', combo)):.2f}, |Mz|={mag(mem_c.max_moment('Mz', combo)):.2f} kN*m")

    bx_top = f"bx_{0}_{0}_{nz-1}"
    if bx_top in m.members:
        mem_b = m.members[bx_top]
        print(f"  Sample roof beam {bx_top}:")
        print(f"    |Mz|={mag(mem_b.max_moment('Mz', combo)):.2f} kN*m, |Fy|={mag(mem_b.max_shear('Fy', combo)):.2f} kN")

    print("=" * 72)
    print(f"Model stats: {nx*ny*nz} nodes, {len(m.members)} members, height ~{N_STORY*STORY_H:.0f} m.")
    print("=" * 72)


if __name__ == "__main__":
    main()
