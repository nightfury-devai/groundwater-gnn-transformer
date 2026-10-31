"""
Phase D3 + D4: Gaussian-kernel adjacency and its symmetric normalisation. No model code.

  D3  A_ij   = exp(-d_ij^2 / sigma^2)         (d in km; diagonal is exp(0) = 1, so NO separate identity is added)
  D4  A_hat  = D^(-1/2) A D^(-1/2),  D = diag(row sums of A)

No hard threshold (CLAUDE.md D3: a smooth kernel instead of an arbitrary "connected / not connected" cut).
sigma is a command-line argument; the default is the std of the off-diagonal pairwise distances.
This script does NOT choose sigma for you: it reports four candidates (std, median, 25 km, 10 km) and
the arrays are built for whatever --sigma you pass (default rule above).

Input  (read only): graph/distance_matrix.npy, graph/well_order.json  (from graph/distance_matrix.py)
Output: graph/adjacency.npy, graph/a_hat.npy, graph/graph_meta.json, graph/d_report.md

Run from the project root:
    .venv/Scripts/python.exe -B -I graph/distance_matrix.py
    .venv/Scripts/python.exe -B -I graph/graph_builder.py [--sigma KM]
"""

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = f"{ROOT}/graph"
EPS = 1e-9
L = []
FAILS = []


def say(s=""):
    print(s)
    L.append(s)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def check(name, cond, detail=""):
    cond = bool(cond)
    if not cond:
        FAILS.append(name)
    say(f"- {'PASS' if cond else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))


def kernel(d, sigma):
    return np.exp(-(d ** 2) / sigma ** 2)


def normalise(A):
    deg = A.sum(axis=1)
    if (deg <= 0).any():
        raise SystemExit("FAIL: a row of A sums to zero, D^(-1/2) undefined")
    inv_sqrt = 1.0 / np.sqrt(deg)
    return A * inv_sqrt[:, None] * inv_sqrt[None, :]


def hand_haversine(lat1, lon1, lat2, lon2, R):
    """Independent pure-Python check (atan2 form of haversine, math module, no numpy)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sigma_stats(d, sigma, names, dists):
    n = len(d)
    off = ~np.eye(n, dtype=bool)
    A = kernel(d, sigma)
    out = {"sigma": sigma}
    for thr in (0.5, 0.1):
        cnt = ((A > thr) & off).sum(axis=1)
        out[f"mean{thr}"], out[f"med{thr}"] = float(cnt.mean()), float(np.median(cnt))
        out[f"zero{thr}"] = int((cnt == 0).sum())
    rs = A.sum(axis=1)
    idx = np.argsort(rs, kind="stable")[:5]
    out["isolated"] = [(names[i], dists[i], float(rs[i])) for i in idx]
    out["cross"] = float(A[off & (np.array(dists)[:, None] != np.array(dists)[None, :])].sum() / A[off].sum())
    ev = np.linalg.eigvalsh(normalise(A))
    out["eig_min"], out["eig_max"] = float(ev[0]), float(ev[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=None,
                    help="kernel width in km; default = std of the off-diagonal pairwise distances")
    args = ap.parse_args()

    d = np.load(f"{G}/distance_matrix.npy")
    with open(f"{G}/well_order.json") as f:
        wo = json.load(f)
    wells = wo["wells"]
    names = [w["well_id"] for w in wells]
    dists = [w["district"] for w in wells]
    lat = [w["lat"] for w in wells]
    lon = [w["lon"] for w in wells]
    R = wo["earth_radius_km"]
    n = len(names)
    assert d.shape == (n, n) == (53, 53)
    off_mask = ~np.eye(n, dtype=bool)
    off = d[off_mask]

    std_off, med_off = float(off.std()), float(np.median(off))
    if args.sigma is None:
        sigma, rule = std_off, "default: std of off-diagonal pairwise distances (population std)"
    else:
        if not (args.sigma > 0 and math.isfinite(args.sigma)):
            raise SystemExit("FAIL: --sigma must be a positive finite number (km)")
        sigma, rule = float(args.sigma), "user-specified --sigma"

    A = kernel(d, sigma)
    A_hat = normalise(A)
    np.save(f"{G}/adjacency.npy", A)
    np.save(f"{G}/a_hat.npy", A_hat)
    with open(f"{G}/graph_meta.json", "w", newline="\n") as f:
        json.dump({"sigma_km": sigma, "sigma_rule": rule, "earth_radius_km": R, "n_nodes": n,
                   "adjacency": "A_ij = exp(-d_ij^2 / sigma^2), diagonal = 1 from the kernel, no threshold",
                   "normalisation": "A_hat = D^(-1/2) A D^(-1/2), D = diag(row sums of A)",
                   "distance_file": "graph/distance_matrix.npy", "well_order_file": "graph/well_order.json",
                   "sigma_candidates_km": {"std_offdiag": std_off, "median_offdiag": med_off, "25km": 25.0, "10km": 10.0}},
                  f, indent=1, sort_keys=True)
        f.write("\n")

    say("# D report: graph construction (distance matrix, adjacency, normalised adjacency)")
    say()
    say("Input: `graph/distance_matrix.npy` and `graph/well_order.json` (written by `graph/distance_matrix.py`). Arrays written: `adjacency.npy`, `a_hat.npy`, `graph_meta.json`. No model code was built.")
    say()
    say("## 1. Node order")
    say()
    say(f"- {n} nodes. Order rule: ascending well_id, the same order `preprocessing/c3_windows.py` iterates the wells (`groupby(sort=False)` on `c2_normalised.csv` sorted by well_id, date). `c3_windows_gap.npz` stores only a per-window `well_id` (42 wells appear there), no 53-node index, so the order was derived from the c3 code and the 42 npz wells were checked to appear in this same relative order (done in `distance_matrix.py`).")
    say(f"- Districts: " + ", ".join(f"{k} {dists.count(k)}" for k in ("Thanjavur", "Nagapattinam", "Thiruvarur")) + ".")
    say(f"- Haversine, R = {R} km. Off-diagonal distance: min {off.min():.4f} km, median {med_off:.3f} km, mean {off.mean():.3f} km, std {std_off:.3f} km, max {off.max():.3f} km.")
    say()
    say("## 2. Sigma used for the saved arrays")
    say()
    say(f"- sigma = **{sigma:.4f} km** ({rule}).")
    say(f"- Candidates (not chosen by the script): std of off-diagonal distances {std_off:.4f} km, median {med_off:.4f} km, 25 km, 10 km.")
    say()

    cands = [("std of off-diagonal distances", std_off), ("median off-diagonal distance", med_off),
             ("25 km", 25.0), ("10 km", 10.0)]
    if not any(abs(sigma - c[1]) < 1e-9 for c in cands):
        cands.append(("sigma used (custom)", sigma))
    st = [(lab, sigma_stats(d, s, names, dists)) for lab, s in cands]

    say("## 3. Sigma candidates")
    say()
    say("Neighbours = other wells j with A_ij above the threshold (off-diagonal, per well). A_ij > 0.5 means d < 0.8326 sigma; A_ij > 0.1 means d < 1.5174 sigma.")
    say()
    rows = []
    for lab, s in st:
        rows.append([lab, f"{s['sigma']:.3f}", f"{s['mean0.5']:.2f} / {s['med0.5']:.1f}", f"{s['mean0.1']:.2f} / {s['med0.1']:.1f}",
                     s["zero0.1"], f"{100 * s['cross']:.1f}%", f"{s['eig_min']:.4f} / {s['eig_max']:.4f}"])
    for line in md_table(["candidate", "sigma (km)", "neighbours A>0.5 (mean / median)", "neighbours A>0.1 (mean / median)",
                          "wells with 0 neighbours at A>0.1", "cross-district share of A weight", "A_hat eigenvalues min / max"], rows):
        say(line)
    say()
    say("### 3a. Five most isolated wells per candidate (smallest row sum of A, diagonal 1 included)")
    say()
    for lab, s in st:
        say(f"**{lab} (sigma {s['sigma']:.3f} km):**")
        say()
        for line in md_table(["rank", "well", "district", "row sum of A"],
                             [[r + 1, w, dn, f"{rs:.4f}"] for r, (w, dn, rs) in enumerate(s["isolated"])]):
            say(line)
        say()

    say("## 4. Checks on the saved arrays (sigma used)")
    say()
    check("distance matrix symmetric", np.array_equal(d, d.T))
    check("distance matrix diagonal = 0", np.all(np.diag(d) == 0))
    check("A symmetric", np.allclose(A, A.T, atol=0, rtol=0))
    check("A diagonal = 1", np.all(np.diag(A) == 1.0))
    check("all A values in (0, 1]", (A > 0).all() and (A <= 1).all(), f"min {A.min():.3e}, max {A.max():.1f}")
    deg = A.sum(axis=1)
    check("D has no zeros", (deg > 0).all(), f"row sums of A: min {deg.min():.4f}, max {deg.max():.4f}")
    check("A_hat symmetric", np.allclose(A_hat, A_hat.T, atol=1e-12, rtol=0), f"max |A_hat - A_hat^T| = {np.abs(A_hat - A_hat.T).max():.2e}")
    ev = np.linalg.eigvalsh((A_hat + A_hat.T) / 2)
    check("eigenvalues of A_hat within [-1, 1]", ev[0] >= -1 - EPS and ev[-1] <= 1 + EPS, f"min {ev[0]:.6f}, max {ev[-1]:.6f}")
    check("A_hat equals D^(-1/2) A D^(-1/2) elementwise", np.allclose(A_hat, A / np.sqrt(np.outer(deg, deg)), atol=1e-14, rtol=0))
    say()

    say("## 5. Hand-calculated haversine spot checks")
    say()
    say("Independent pure-Python haversine (atan2 form, `math` module) against the matrix entry.")
    say()
    iu = np.triu_indices(n, 1)
    dv = d[iu]
    nz = np.flatnonzero(dv > 0)
    i_close = nz[np.argmin(dv[nz])]
    i_far = int(np.argmax(dv))
    cross = np.array([dists[a] != dists[b] for a, b in zip(*iu)])
    i_cross = np.flatnonzero(cross)[np.argmin(dv[cross])]
    spots = [("closest non-zero pair", i_close, {"Paramasivapuram II", "Paramasivapuram III"}, 0.15, 0.05),
             ("farthest pair", i_far, {"Peravurani", "Puttur"}, 125.3, 0.5),
             ("closest cross-district pair", i_cross, None, None, None)]
    rows = []
    for lab, k, exp_names, exp_d, tol in spots:
        a, b = int(iu[0][k]), int(iu[1][k])
        hand = hand_haversine(lat[a], lon[a], lat[b], lon[b], R)
        agree = abs(hand - d[a, b]) < 1e-9
        check(f"{lab}: matrix = hand haversine", agree, f"{names[a]} - {names[b]}: matrix {d[a, b]:.6f} km, hand {hand:.6f} km")
        if exp_names is not None:
            check(f"{lab}: expected wells and distance", {names[a], names[b]} == exp_names and abs(d[a, b] - exp_d) <= tol,
                  f"expected {sorted(exp_names)} about {exp_d} km; found {names[a]} / {names[b]} at {d[a, b]:.3f} km")
        rows.append([lab, f"{names[a]} ({dists[a]})", f"{names[b]} ({dists[b]})", f"{d[a, b]:.4f}", f"{hand:.4f}", f"{A[a, b]:.6f}"])
    say()
    for line in md_table(["pair", "well 1 (district)", "well 2 (district)", "matrix d (km)", "hand d (km)", "A_ij (sigma used)"], rows):
        say(line)
    say()

    say("## 6. Pairs at distance exactly 0 (kept as separate nodes, not merged, dropped or jittered)")
    say()
    z = [(int(a), int(b)) for a, b in zip(*iu) if d[a, b] == 0]
    if z:
        for line in md_table(["well 1", "well 2", "district(s)", "lat", "lon", "d (km)", "A_ij"],
                             [[names[a], names[b], dists[a] if dists[a] == dists[b] else f"{dists[a]} / {dists[b]}",
                               f"{lat[a]:.6f}", f"{lon[a]:.6f}", "0", f"{A[a, b]:.1f}"] for a, b in z]):
            say(line)
    else:
        say("- None.")
    say()
    say(f"- {len(z)} pair(s). Expected Icchankottai / Icchankottai1 and Vedaranyam / Vedaranyam1: "
        + ("matches." if {frozenset((names[a], names[b])) for a, b in z} == {frozenset(("Icchankottai", "Icchankottai1")), frozenset(("Vedaranyam", "Vedaranyam1"))} else "DOES NOT MATCH."))
    say()

    say("## 7. Cross-district share of off-diagonal A weight (sigma used)")
    say()
    diff = np.array(dists)[:, None] != np.array(dists)[None, :]
    tot = A[off_mask].sum()
    share = A[off_mask & diff].sum() / tot
    say(f"- Total off-diagonal A weight {tot:.3f}; between wells of different districts {A[off_mask & diff].sum():.3f}; **share {100 * share:.2f}%**.")
    say()
    rows = []
    for a_ in ("Thanjavur", "Nagapattinam", "Thiruvarur"):
        rows.append([a_] + [f"{A[np.ix_([i for i in range(n) if dists[i] == a_], [j for j in range(n) if dists[j] == b_])].sum() - (A.diagonal()[[i for i in range(n) if dists[i] == a_]].sum() if a_ == b_ else 0):.3f}" for b_ in ("Thanjavur", "Nagapattinam", "Thiruvarur")])
    for line in md_table(["off-diagonal A weight, row district to column district", "Thanjavur", "Nagapattinam", "Thiruvarur"], rows):
        say(line)
    say()
    say("## 8. Files")
    say()
    say("- `graph/distance_matrix.py`, `graph/graph_builder.py`, `graph/distance_matrix.npy`, `graph/well_order.json`, `graph/adjacency.npy`, `graph/a_hat.npy`, `graph/graph_meta.json`, `graph/d_report.md`.")
    say()
    say(f"Overall: {'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")

    with open(f"{G}/d_report.md", "w", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    if FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
