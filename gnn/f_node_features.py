"""
Phase F1: GCN node features for every window (last observation carried forward, LOCF). Nothing is trained.

Design (final, decided 2026-10-25). For window w with target well i and time t = last_input_date[w],
EVERY well j of the 53-node graph (order = graph/well_order.json, never recomputed) gets 8 values:
  - the 6 normalised features of the LATEST reading of well j with date <= t   (same 6 columns, same order as X in C3)
  - staleness = (t - date of that reading) in days / 91.25 (quarters), then z-scored with the mean and std
    over the TRAIN windows only (observed nodes only), saved in gnn/f_staleness_scaler.json
  - has_reading = 1 if the well has any reading with date <= t, else 0
A well with no reading <= t gets zeros for all 8 values (features 0, staleness 0, has_reading 0).
Only readings dated on or before t are ever used: nothing after t can reach a node (no leakage).

Input  (read only): preprocessing/c3_windows_gap.npz, preprocessing/c2_normalised.csv,
                    preprocessing/c3_windows.py (feature order), graph/well_order.json
Output: gnn/f_node_features.npz (NF, node_idx, well_id, split, last_input_date), gnn/f_staleness_scaler.json

Standalone:  .venv\\Scripts\\python.exe -B -I gnn\\f_node_features.py      (normally called by gnn\\run_phase_f.py)
numpy and the csv module only.
"""

import ast
import csv
import json
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_IN = f"{ROOT}/preprocessing/c3_windows_gap.npz"
C2 = f"{ROOT}/preprocessing/c2_normalised.csv"
C3_PY = f"{ROOT}/preprocessing/c3_windows.py"
ORDER = f"{ROOT}/graph/well_order.json"
OUT_NPZ = f"{ROOT}/gnn/f_node_features.npz"
OUT_SC = f"{ROOT}/gnn/f_staleness_scaler.json"
DAYS_PER_QUARTER = 91.25
SPLITS = ("train", "val", "test")


def need(path):
    if not os.path.exists(path):
        raise SystemExit(f"FAIL: required input file is missing: {path}")


def feature_order_from_c3():
    """Read FEATURES = [...] from c3_windows.py so the z-column order is the one X was built with."""
    with open(C3_PY, encoding="utf8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "FEATURES" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise SystemExit("FAIL: FEATURES not found in preprocessing/c3_windows.py")


def save_npz(path, arrays):
    """Deterministic .npz (np.savez stamps the current time into the zip, so files would differ per run)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            with zf.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.ascontiguousarray(arrays[name]), allow_pickle=False)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def build():
    """Build the node features, run the F1 checks. Returns (report_lines, results_dict, checks) where
    checks = [(name, ok, detail), ...]."""
    for p in (NPZ_IN, C2, C3_PY, ORDER):
        need(p)
    L, checks, res = [], [], {}

    def say(s=""):
        L.append(s)

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        say(f"- {'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))

    z = np.load(NPZ_IN, allow_pickle=False)
    X, wid, split, last = z["X"], z["well_id"].astype(str), z["split"].astype(str), z["last_input_date"].astype(str)
    W, Lw, F = X.shape
    with open(ORDER) as f:
        wells = json.load(f)["wells"]
    order = [w["well_id"] for w in wells]
    n_nodes = len(order)
    assert [w["index"] for w in wells] == list(range(n_nodes))

    feats = feature_order_from_c3()
    zcols = ["z_" + c for c in feats]
    assert len(feats) == F, f"X has {F} features but c3_windows.py lists {len(feats)}"

    # ---- readings per well (after C0 dedup), z-columns in the X order, sorted by date
    per = {w: ([], []) for w in order}
    with open(C2, newline="", encoding="utf8") as f:
        rd = csv.DictReader(f)
        missing = [c for c in zcols + ["well_id", "date"] if c not in rd.fieldnames]
        if missing:
            raise SystemExit(f"FAIL: c2_normalised.csv lacks columns {missing}")
        for r in rd:
            if r["well_id"] not in per:
                raise SystemExit(f"FAIL: well {r['well_id']} in c2_normalised.csv is not in graph/well_order.json")
            per[r["well_id"]][0].append(r["date"])
            per[r["well_id"]][1].append([float(r[c]) for c in zcols])
    dates, Zr = {}, {}
    for w in order:
        d = np.array(per[w][0], dtype="datetime64[D]")
        o = np.argsort(d, kind="stable")
        dates[w], Zr[w] = d[o], np.array(per[w][1], dtype=np.float64).reshape(-1, F)[o]
    n_readings = sum(len(dates[w]) for w in order)

    # ---- LOCF node features
    t = last.astype("datetime64[D]")
    feat = np.zeros((W, n_nodes, F), dtype=np.float64)
    stale_q = np.zeros((W, n_nodes), dtype=np.float64)        # staleness in quarters, before z-scoring
    has = np.zeros((W, n_nodes), dtype=np.float64)
    used_lag_days = np.full((W, n_nodes), np.nan)             # date used - t, in days (must be <= 0)
    used_idx = np.full((W, n_nodes), -1, dtype=np.int64)
    for j, w in enumerate(order):
        idx = np.searchsorted(dates[w], t, side="right") - 1  # latest reading with date <= t
        obs = idx >= 0
        used_idx[:, j] = idx
        feat[obs, j, :] = Zr[w][idx[obs]]
        lag = (dates[w][idx[obs]] - t[obs]).astype(float)
        used_lag_days[obs, j] = lag
        stale_q[obs, j] = np.abs(lag) / DAYS_PER_QUARTER      # lag <= 0 by construction (checked below); abs avoids -0.0
        has[obs, j] = 1.0

    # ---- staleness z-score: TRAIN windows only, observed nodes only
    tr = split == "train"
    vals = stale_q[tr][has[tr] == 1]
    mean, std = float(vals.mean()), float(vals.std())
    std_used = std if std > 0 else 1.0
    stale_z = np.where(has == 1, (stale_q - mean) / std_used, 0.0)
    with open(OUT_SC, "w", newline="\n") as f:
        json.dump({"mean": mean, "std": std, "std_used": std_used, "n_values": int(vals.size),
                   "unit": "quarters (days / 91.25)", "fit": "TRAIN windows only, observed nodes only",
                   "n_train_windows": int(tr.sum())}, f, indent=1, sort_keys=True)
        f.write("\n")

    NF = np.concatenate([feat, stale_z[..., None], has[..., None]], axis=2).astype(np.float32)   # (W, 53, 8)
    node_idx = np.array([order.index(w) for w in wid], dtype=np.int64)
    save_npz(OUT_NPZ, {"NF": NF, "node_idx": node_idx, "well_id": wid, "split": split, "last_input_date": last})

    # ---- report + checks
    say("## F1. Node features (last observation carried forward)")
    say()
    say(f"- Windows {W}, nodes {n_nodes} (from `graph/well_order.json`), readings used from `c2_normalised.csv`: {n_readings}. "
        f"Feature order read from `preprocessing/c3_windows.py`: {', '.join(feats)}.")
    say(f"- Node vector = {F} features + staleness + has_reading = {NF.shape[2]} values. "
        f"Staleness = days since the carried reading / {DAYS_PER_QUARTER}, then z-scored with train-only mean {mean:.4f} and std {std:.4f} "
        f"(over {vals.size} observed node values in {int(tr.sum())} train windows), saved in `gnn/f_staleness_scaler.json`.")
    say("- Output: `gnn/f_node_features.npz` with NF, node_idx, well_id, split, last_input_date.")
    say()

    check("NF shape is (windows, nodes, 8)", NF.shape == (W, n_nodes, F + 2), str(NF.shape))
    check("NF has no NaN", not np.isnan(NF).any(), f"{int(np.isnan(NF).sum())} NaN")

    # 2. no reading after t, checked two ways: the lag of the reading used, and an independent brute-force search
    max_lag = float(np.nanmax(used_lag_days))
    check("every reading used has date <= last_input_date", max_lag <= 0, f"max (date used - t) = {max_lag:.0f} days")
    bad = 0
    for wi in range(W):
        for j, w in enumerate(order):
            ok = np.flatnonzero(dates[w] <= t[wi])
            ref = ok[-1] if ok.size else -1
            bad += int(ref != used_idx[wi, j])
    check("independent brute-force search finds the same reading for all window-node pairs", bad == 0, f"{bad} differences in {W * n_nodes} pairs")
    check("unobserved nodes are all-zero (features, staleness, has_reading)",
          bool((NF[has == 0] == 0).all()), f"{int((has == 0).sum())} unobserved window-node pairs")

    # 3. target well's own row equals the last window position
    tgt_feat = NF[np.arange(W), node_idx, :F].astype(np.float64)
    dmax = float(np.abs(tgt_feat - X[:, Lw - 1, :]).max())
    check("NF[w, node_idx[w], :6] equals X[w, last position, :]", dmax < 1e-6, f"max abs difference {dmax:.2e} (float32 storage)")
    tgt_has = NF[np.arange(W), node_idx, F + 1]
    check("target well is observed with staleness 0 in every window", bool((tgt_has == 1).all() and (stale_q[np.arange(W), node_idx] == 0).all()))

    # 6. node_idx mapping
    ok_map = all(order[node_idx[i]] == wid[i] for i in range(W))
    check("node_idx maps every window's well_id to its position in well_order.json", ok_map)

    # 4. observed nodes per window and staleness per split
    say()
    say("Observed nodes per window (has_reading = 1) and staleness of the carried readings, by split:")
    say()
    rows = []
    for s in SPLITS:
        m = split == s
        n_obs = has[m].sum(axis=1)
        sq = stale_q[m][has[m] == 1]
        rows.append([s, int(m.sum()), f"{n_obs.min():.0f} / {np.median(n_obs):.0f} / {n_obs.max():.0f}",
                     f"{sq.min():.2f} / {np.median(sq):.2f} / {sq.max():.2f}"])
    res["observed_per_split"] = {r[0]: r[2] for r in rows}
    res["staleness_quarters_per_split"] = {r[0]: r[3] for r in rows}
    for line in md_table(["split", "windows", "observed nodes per window (min / median / max)", "staleness in quarters (min / median / max, observed nodes)"], rows):
        say(line)
    say()

    # 5. wells without a window
    with_window = set(wid.tolist())
    no_win = [w for w in order if w not in with_window]
    say(f"Wells with no window of their own: {len(no_win)} of {n_nodes}. Number of windows in which each is still observed as a neighbour node:")
    say()
    rows = []
    for w in no_win:
        j = order.index(w)
        n_ob = int(has[:, j].sum())
        rows.append([w, wells[j]["district"], len(dates[w]), n_ob, f"{100 * n_ob / W:.1f}%"])
    for line in md_table(["well", "district", "readings", "windows in which observed", "share of windows"], rows):
        say(line)
    say()
    res.update({"n_windows": int(W), "n_nodes": int(n_nodes), "max_date_used_minus_t_days": max_lag,
                "target_row_max_abs_diff": dmax, "staleness_scaler": {"mean": mean, "std": std, "n": int(vals.size)},
                "wells_without_window": {w: int(has[:, order.index(w)].sum()) for w in no_win},
                "split_counts": {s: int((split == s).sum()) for s in SPLITS}})
    check("each well without a window is read from the table (count taken from the files)", len(no_win) == n_nodes - len(with_window), f"{len(no_win)} wells")
    return L, res, checks


if __name__ == "__main__":
    lines, results, chk = build()
    print("\n".join(lines))
    fails = [n for n, ok, _ in chk if not ok]
    print("\n" + ("ALL CHECKS PASSED" if not fails else "FAILED: " + "; ".join(fails)))
    sys.exit(1 if fails else 0)
