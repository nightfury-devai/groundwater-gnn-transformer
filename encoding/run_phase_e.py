"""
Helper: run Phase E and print the key numbers. Changes no existing script or data.

    .venv\\Scripts\\python.exe -B -I encoding\\run_phase_e.py [e1 | e2 | e3 | all]      (default: all)

encoding/e_temporal.py always runs completely (E3 needs the windows from E1 and the table from E2); the
argument only selects which sections are printed. Any value that differs from what is expected is printed
with MISMATCH. Standard library only (numpy and torch are used inside e_temporal.py).
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENC = os.path.join(ROOT, "encoding")
MISMATCHES = []


def show(label, value, ok=True, expect=None):
    tag = ""
    if not ok:
        tag = f"   MISMATCH (expected {expect})"
        MISMATCHES.append(label)
    print(f"  {label}: {value}{tag}")


def run_script():
    print("running encoding/e_temporal.py ...")
    r = subprocess.run([sys.executable, "-B", "-I", os.path.join(ENC, "e_temporal.py")], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        # a failed CHECK still writes the report; a crash does not. Show everything so nothing is hidden.
        print(f"e_temporal.py exited with code {r.returncode}")
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
    else:
        print("encoding/e_temporal.py finished (exit code 0)")
    return r.returncode


def load_results():
    path = os.path.join(ENC, "e_results.json")
    if not os.path.exists(path):
        print("FAILED: encoding/e_results.json was not written (see the error above)")
        sys.exit(1)
    return json.load(open(path))


def e1(res):
    print("\n=== E1 ===")
    r = res["E1"]
    show("X shape", tuple(r["X_shape"]), tuple(r["X_shape"]) == (840, 8, 6), (840, 8, 6))
    c = r["split_counts"]
    show("split counts train / val / test", f"{c['train']} / {c['val']} / {c['test']}", (c["train"], c["val"], c["test"]) == (581, 147, 112), "581 / 147 / 112")
    show("wells", r["n_wells"], r["n_wells"] == 42, 42)
    show("NaN in X", r["nan_X"], r["nan_X"] == 0, 0)
    show("NaN in G (total)", r["nan_G_total"], r["nan_G_total"] == 42, 42)
    show("NaN in G[:,0]", r["nan_G_col0"], r["nan_G_col0"] == 42, 42)
    show("NaN in G[:,1:]", r["nan_G_rest"], r["nan_G_rest"] == 0, 0)
    print(f"  X range: {r['X_min']:.3f} to {r['X_max']:.3f}   (gap information G is stored but not used by Phase E)")


def e2(res):
    print("\n=== E2 ===")
    r = res["E2"]
    show("PE table shape", tuple(r["pe_shape"]), tuple(r["pe_shape"]) == (8, 64), (8, 64))
    show("PE[0,0] and PE[0,1]", f"{r['pe00']} and {r['pe01']}", r["pe00"] == 0.0 and r["pe01"] == 1.0, "0 and 1")
    show("PE min / max", f"{r['pe_min']:.4f} / {r['pe_max']:.4f}", r["pe_min"] >= -1 and r["pe_max"] <= 1, "within [-1, 1]")
    show("8 rows all different", r["rows_distinct"], r["rows_distinct"], True)
    for p, c, tab, hand in r["hand_entries"]:
        show(f"PE[{p},{c}] table vs hand", f"{tab:.10f} vs {hand:.10f}", abs(tab - hand) < 1e-12, "equal")


def e3(res):
    print("\n=== E3 ===")
    r = res["E3"]
    if not r["torch_ok"]:
        print(f"  BLOCKED: import torch failed: {r['torch_error']}")
        print("  (nothing was installed; E1 and E2 are complete)")
        return
    print(f"  torch {r['torch_version']}, CUDA available: {r['cuda']}")
    show("output shape", tuple(r["output_shape"]), tuple(r["output_shape"]) == (4, 8, 64), (4, 8, 64))
    show("parameter count", r["n_params"], r["n_params"] == 448, 448)
    show("max abs difference to hand calculation", f"{r['max_abs_err']:.2e}", r["max_abs_err"] < 1e-5, "< 1e-5")


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if mode not in ("e1", "e2", "e3", "all"):
        print(f"unknown option '{mode}'. Use: e1, e2, e3 or all (default).")
        sys.exit(2)
    code = run_script()
    res = load_results()
    if mode in ("e1", "all"):
        e1(res)
    if mode in ("e2", "all"):
        e2(res)
    if mode in ("e3", "all"):
        e3(res)
    print("\n=== report ===")
    with open(os.path.join(ENC, "e_report.md"), encoding="utf8") as f:
        last = [ln.strip() for ln in f if ln.strip()][-1]
    show("last line of encoding/e_report.md", last, last == "ALL CHECKS PASSED", "ALL CHECKS PASSED")
    print("\n" + ("ALL EXPECTED VALUES MATCH" if not MISMATCHES else "MISMATCH in: " + "; ".join(MISMATCHES)))
    sys.exit(1 if (MISMATCHES or code) else 0)


if __name__ == "__main__":
    main()
