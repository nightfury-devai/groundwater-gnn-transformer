"""
Helper: run Phase D and print a plain summary of what it produced. Changes no existing script or data.

    .venv\\Scripts\\python.exe -B -I graph\\run_phase_d.py [d1d2 | d3d4 | all]      (default: all)

d1d2 : runs graph/distance_matrix.py, then prints D1 (wells) and D2 (distance matrix).
d3d4 : runs graph/graph_builder.py (default sigma), then prints D3 (adjacency) and D4 (normalised adjacency).
Every expected value that does not match is printed with MISMATCH next to it.
Only numpy and the standard library are used.
"""

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH = os.path.join(ROOT, "graph")
MISMATCHES = []


def run_script(name):
    """Run a graph/ script with the same interpreter and -B -I; stop with the error text if it fails."""
    print(f"running graph/{name} ...")
    r = subprocess.run([sys.executable, "-B", "-I", os.path.join(GRAPH, name)], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED: graph/{name} exited with code {r.returncode}")
        print(r.stdout)
        print(r.stderr)
        sys.exit(r.returncode or 1)
    print(f"graph/{name} finished (exit code 0)")


def show(label, value, ok=None, expect=None):
    """Print one line; ok=False adds MISMATCH (and the expectation) so it cannot be missed."""
    tag = ""
    if ok is False:
        tag = f"   MISMATCH (expected {expect})"
        MISMATCHES.append(label)
    print(f"  {label}: {value}{tag}")


def load(name):
    path = os.path.join(GRAPH, name)
    if not os.path.exists(path):
        print(f"MISSING: graph/{name} (run d1d2 / all first)")
        sys.exit(1)
    return np.load(path) if name.endswith(".npy") else json.load(open(path))


def d1():
    print("\n=== D1 ===")
    wells = load("well_order.json")["wells"]
    show("number of wells", len(wells), len(wells) == 53, 53)
    for w in wells[:3] + wells[-1:]:
        print(f"  [{w['index']:2d}] {w['well_id']:<22} lat {w['lat']:.6f}  lon {w['lon']:.6f}  {w['district']}")


def d2():
    print("\n=== D2 ===")
    d = load("distance_matrix.npy")
    n = d.shape[0]
    show("distance_matrix.npy shape", d.shape, d.shape == (53, 53), (53, 53))
    show("diagonal all zero", bool(np.all(np.diag(d) == 0)), bool(np.all(np.diag(d) == 0)), True)
    sym = bool(np.array_equal(d, d.T))
    show("symmetric", sym, sym, True)
    wells = load("well_order.json")["wells"]
    nm = [w["well_id"] for w in wells]
    v02, v20 = d[0, 2], d[2, 0]
    show(f"d[0,2] ({nm[0]} to {nm[2]}) km", f"{v02:.4f}", abs(v02 - 35.47) <= 0.05, "about 35.47")
    show(f"d[2,0] ({nm[2]} to {nm[0]}) km", f"{v20:.4f}", abs(v20 - 35.47) <= 0.05, "about 35.47")
    off = d[~np.eye(n, dtype=bool)]
    print(f"  off-diagonal km: min {off.min():.4f}   median {np.median(off):.4f}   max {off.max():.4f}")
    pairs = n * (n - 1) // 2
    show("unique pairs", pairs, pairs == 1378, 1378)


def d3():
    print("\n=== D3 ===")
    meta = load("graph_meta.json")
    A = load("adjacency.npy")
    show("sigma used (km)", f"{meta['sigma_km']:.4f}  [{meta['sigma_rule']}]", abs(meta["sigma_km"] - 24.7975) < 1e-3, "24.7975 (std of off-diagonal distances)")
    show("A shape", A.shape, A.shape == (53, 53), (53, 53))
    diag1 = bool(np.all(np.diag(A) == 1.0))
    show("A diagonal all 1", diag1, diag1, True)
    sym = bool(np.allclose(A, A.T, atol=0, rtol=0))
    show("A symmetric", sym, sym, True)
    print(f"  A min {A.min():.6e}   A max {A.max():.6f}")


def d4():
    print("\n=== D4 ===")
    Ah = load("a_hat.npy")
    rs = Ah.sum(axis=1)
    show("A_hat shape", Ah.shape, Ah.shape == (53, 53), (53, 53))
    print(f"  A_hat row sum: min {rs.min():.6f}   max {rs.max():.6f}")
    with open(os.path.join(GRAPH, "d_report.md"), encoding="utf8") as f:
        last = [ln.strip() for ln in f if ln.strip()][-1]
    ok = "ALL CHECKS PASSED" in last
    show("last line of graph/d_report.md", last, ok, "ALL CHECKS PASSED")


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if mode not in ("d1d2", "d3d4", "all"):
        print(f"unknown option '{mode}'. Use: d1d2, d3d4 or all (default).")
        sys.exit(2)
    if mode in ("d1d2", "all"):
        run_script("distance_matrix.py")
        d1()
        d2()
    if mode in ("d3d4", "all"):
        run_script("graph_builder.py")
        d3()
        d4()
    print("\n" + ("ALL EXPECTED VALUES MATCH" if not MISMATCHES else "MISMATCH in: " + "; ".join(MISMATCHES)))
    sys.exit(1 if MISMATCHES else 0)


if __name__ == "__main__":
    main()
