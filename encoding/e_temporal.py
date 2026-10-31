"""
Phase E: temporal encoding of each well's 8-reading window. Nothing is trained.

  E1  per-well sequence X_i of shape [L x F] = [8 x 6]  (the standardised C2/C3 windows)
  E2  sinusoidal positional encoding, PE table of shape (8, 64), numpy only
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_h))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_h)),   d_h = 64, pos = 0..7
  E3  linear embedding to d_h = 64, THEN add the positional encoding:
        H0 = (X W_e + b_e) + PE
      Order note: CLAUDE.md lists E2 before E3, but PE has dimension d_h = 64 while X has only 6
      features, so PE can only be added after the embedding has projected X to 64 dimensions.

Gap information (G, quarters between readings) is stored in the windows but NOT used here.

Input  (read only): preprocessing/c3_windows_gap.npz, preprocessing/c4_split_dates.json
Output: encoding/pe_table.npy, encoding/e_report.md, encoding/e_results.json (key numbers for run_phase_e.py)
E3 needs torch. If torch cannot be imported the exact error is recorded, nothing is installed, and E3 is
marked BLOCKED (E1 and E2 are still completed).

Run from the project root:  .venv/Scripts/python.exe -B -I encoding/e_temporal.py
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = f"{ROOT}/preprocessing/c3_windows_gap.npz"
SPLIT = f"{ROOT}/preprocessing/c4_split_dates.json"
OUT = f"{ROOT}/encoding"
L_WIN, F_IN, D_H = 8, 6, 64
EXPECT_SPLIT = {"train": 581, "val": 147, "test": 112}

L = []
RES = {}
STATUS = []           # (name, "PASS" | "FAIL" | "BLOCKED")


def say(s=""):
    print(s)
    L.append(s)


def check(name, cond, detail=""):
    st = "PASS" if cond else "FAIL"
    STATUS.append((name, st))
    say(f"- {st}: {name}" + (f" ({detail})" if detail else ""))
    return bool(cond)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def positional_encoding(length, d):
    """Sinusoidal PE table (length, d); column 2i is sin, column 2i+1 is cos, both with frequency index i."""
    if d % 2:
        raise ValueError("d_h must be even")
    pe = np.zeros((length, d))
    pos = np.arange(length)[:, None]
    i = np.arange(d // 2)[None, :]
    angle = pos / np.power(10000.0, 2 * i / d)
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


def main():
    os.makedirs(OUT, exist_ok=True)
    say("# E report: temporal encoding (E1 sequences, E2 positional encoding, E3 embedding + PE)")
    say()
    say("Nothing is trained. Gap information (quarters between readings, array G) is stored in the windows but is NOT used by this encoding yet.")
    say()

    # ------------------------------------------------------------------ E1
    say("## E1. Per-well sequences X_i of shape [L x F]")
    say()
    z = np.load(NPZ, allow_pickle=False)
    say("Keys and shapes of `preprocessing/c3_windows_gap.npz`:")
    say()
    for line in md_table(["key", "shape", "dtype"], [[k, z[k].shape, z[k].dtype] for k in z.files]):
        say(line)
    say()
    with open(SPLIT) as f:
        cut = json.load(f)
    X, G, y, split, wid = z["X"], z["G"], z["y"], z["split"], z["well_id"]
    td = z["target_date"]
    counts = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    n_wells = len(set(wid.tolist()))
    nan_x = int(np.isnan(X).sum())
    nan_g_all = int(np.isnan(G).sum())
    nan_g0 = int(np.isnan(G[:, 0]).sum())
    nan_g_rest = int(np.isnan(G[:, 1:]).sum())
    RES["E1"] = {"X_shape": list(X.shape), "y_shape": list(y.shape), "split_counts": counts, "n_wells": n_wells,
                 "nan_X": nan_x, "nan_G_total": nan_g_all, "nan_G_col0": nan_g0, "nan_G_rest": nan_g_rest,
                 "X_min": float(X.min()), "X_max": float(X.max())}
    check("X shape is (840, 8, 6)", X.shape == (840, L_WIN, F_IN), str(X.shape))
    check("target y has 840 values and y_metres matches", y.shape == (840,) and z["y_metres"].shape == (840,))
    check("split counts 581 train / 147 val / 112 test", counts == EXPECT_SPLIT, str(counts))
    check("42 wells", n_wells == 42, str(n_wells))
    check("no NaN in X (inputs)", nan_x == 0, f"{nan_x} NaN")
    check("G[:,0] is NaN in 42 windows", nan_g0 == 42, f"{nan_g0} NaN")
    check("G[:,1:] has no NaN (all NaN in G are in column 0)", nan_g_rest == 0 and nan_g_all == nan_g0, f"total {nan_g_all}, column 0 {nan_g0}, columns 1-7 {nan_g_rest}")
    check("X is finite", bool(np.isfinite(X).all()), f"min {X.min():.3f}, max {X.max():.3f}")
    # the split labels must follow the C4 cutoffs (guards against an out-of-date npz)
    # ISO date strings sort correctly as text; numpy has no max/min for string arrays, so use Python's
    tds = {s: td[split == s].tolist() for s in ("train", "val", "test")}
    ok_split = (max(tds["train"]) <= cut["train_cutoff"] < min(tds["val"])
                and max(tds["val"]) <= cut["val_cutoff"] < min(tds["test"]))
    check("split labels agree with c4_split_dates.json cutoffs", bool(ok_split), f"train <= {cut['train_cutoff']}, val <= {cut['val_cutoff']}, test after")
    say()

    # ------------------------------------------------------------------ E2
    say("## E2. Sinusoidal positional encoding (numpy only)")
    say()
    pe = positional_encoding(L_WIN, D_H)
    np.save(f"{OUT}/pe_table.npy", pe)
    rows_distinct = len({tuple(r) for r in pe.tolist()}) == L_WIN
    # three entries recomputed independently with math.sin / math.cos
    hand = [(1, 0, math.sin(1 / 10000 ** (0 / D_H))),
            (3, 11, math.cos(3 / 10000 ** (2 * 5 / D_H))),
            (7, 62, math.sin(7 / 10000 ** (2 * 31 / D_H)))]
    hand_ok = all(abs(pe[p, c] - v) < 1e-12 for p, c, v in hand)
    RES["E2"] = {"pe_shape": list(pe.shape), "pe_min": float(pe.min()), "pe_max": float(pe.max()),
                 "pe00": float(pe[0, 0]), "pe01": float(pe[0, 1]), "rows_distinct": bool(rows_distinct),
                 "hand_entries": [[p, c, float(pe[p, c]), float(v)] for p, c, v in hand]}
    check("PE table shape is (8, 64), saved to encoding/pe_table.npy", pe.shape == (L_WIN, D_H), str(pe.shape))
    check("PE[0,0] = 0 and PE[0,1] = 1", pe[0, 0] == 0.0 and pe[0, 1] == 1.0, f"{pe[0, 0]}, {pe[0, 1]}")
    check("all PE values in [-1, 1]", bool(pe.min() >= -1 and pe.max() <= 1), f"min {pe.min():.4f}, max {pe.max():.4f}")
    check("all 8 rows are different", rows_distinct)
    check("3 entries match independent math.sin / math.cos", hand_ok,
          "; ".join(f"PE[{p},{c}] table {pe[p, c]:.10f} vs hand {v:.10f}" for p, c, v in hand))
    say()

    # ------------------------------------------------------------------ E3
    say("## E3. Linear embedding to d_h = 64, then add PE: H0 = (X W_e + b_e) + PE")
    say()
    try:
        import torch
        import torch.nn as nn
        tv, cuda = torch.__version__, bool(torch.cuda.is_available())
        torch_err = None
    except Exception as e:                                    # any import failure (missing package, blocked DLL)
        tv, cuda, torch_err = None, None, f"{type(e).__name__}: {e}"
    RES["E3"] = {"torch_ok": torch_err is None, "torch_version": tv, "cuda": cuda, "torch_error": torch_err}
    if torch_err is not None:
        say(f"- `import torch` failed with the exact error: `{torch_err}`")
        say("- Nothing was installed. E3 is **BLOCKED**; E1 and E2 are complete. Fallback in CLAUDE.md: Colab or Kaggle GPU, or fix the torch install and re-run.")
        STATUS.append(("E3 embedding + PE (needs torch)", "BLOCKED"))
    else:
        say(f"- torch {tv}, CUDA available: {cuda}")
        torch.manual_seed(42)
        lin = nn.Linear(F_IN, D_H)
        tr_idx = np.flatnonzero(split == "train")[:4]
        xb = torch.tensor(X[tr_idx], dtype=torch.float32)                 # (4, 8, 6)
        pe_t = torch.tensor(pe, dtype=torch.float32)                      # (8, 64)
        with torch.no_grad():
            h0 = lin(xb) + pe_t                                           # broadcast over the batch
        n_par = sum(p.numel() for p in lin.parameters())
        W = lin.weight.detach().double().numpy()                          # (64, 6)
        b = lin.bias.detach().double().numpy()
        by_hand = (X[tr_idx] @ W.T + b) + pe                              # float64 reference
        err = float(np.abs(h0.numpy().astype(np.float64) - by_hand).max())
        RES["E3"].update({"output_shape": list(h0.shape), "n_params": int(n_par), "max_abs_err": err,
                          "first4_train_idx": tr_idx.tolist()})
        check("output shape is (4, 8, 64)", tuple(h0.shape) == (4, L_WIN, D_H), str(tuple(h0.shape)))
        check("parameter count 6*64 + 64 = 448", n_par == F_IN * D_H + D_H, str(n_par))
        check("output equals (X @ W^T + b) + PE by hand within 1e-5", err < 1e-5, f"max abs diff {err:.2e}")
    say()

    # ------------------------------------------------------------------ summary
    fails = [n for n, s in STATUS if s == "FAIL"]
    blocked = [n for n, s in STATUS if s == "BLOCKED"]
    say("## Gap information")
    say()
    say("- The gap array G (quarters between consecutive readings, NaN in column 0 for 42 windows) is stored in the windows but is not used by this encoding yet.")
    say()
    say("## Files")
    say()
    say("- `encoding/e_temporal.py`, `encoding/pe_table.npy`, `encoding/e_report.md`, `encoding/e_results.json`, `encoding/run_phase_e.py`.")
    say()
    if fails:
        final = "FAILED: " + "; ".join(fails)
    elif blocked:
        final = "E1 and E2: ALL CHECKS PASSED. E3: BLOCKED (" + RES["E3"]["torch_error"] + ")"
    else:
        final = "ALL CHECKS PASSED"
    say(final)
    RES["final_line"] = final
    with open(f"{OUT}/e_report.md", "w", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    with open(f"{OUT}/e_results.json", "w", newline="\n") as f:
        json.dump(RES, f, indent=1, sort_keys=True)
        f.write("\n")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
