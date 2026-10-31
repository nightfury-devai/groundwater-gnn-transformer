"""
Phase C3: sliding windows of 8 readings; target = depth of the reading after the window.

A window ends at reading k of a well and holds that well's readings k-7..k (8 readings). The target
is the NEXT reading's depth (next_depth, from C1). A window never mixes wells and needs at least 8
readings up to k in its well. Two variants:
  GAP    : the 8 most recent readings of the well, no constraint on the gaps between them.
           The gap (in quarters) before every position is stored in G; X keeps exactly the 6 features.
  STRICT : the 8 readings lie in 8 consecutive quarters (all 7 links inside the window have gap 1)
           AND the target is the very next quarter (target gap = 1), i.e. 9 consecutive quarters
           (decision 2026-10-18). The count without the target-gap rule is reported for reference.
Split: by TARGET date with the two cutoffs from C4. Nothing is shuffled: windows are ordered by well
and by time inside each well. No value is filled or interpolated.

G[:, 0] is the gap from the reading BEFORE the window to the first window reading. It is NaN when
the window starts at the very first reading of the well (nothing before it); G[:, 1:] is always
defined. This is kept as NaN on purpose (a made-up number would be a filled value).

Input  (read only): preprocessing/c2_normalised.csv, preprocessing/c2_scaler.json,
                    preprocessing/c4_split_dates.json, data/cgwb_*.csv (well -> district only)
Output: preprocessing/c3_windows_gap.npz, c3_windows_strict.npz, c3_report.md
npz arrays: X (N,8,6, normalised), G (N,8), target_gap, y (normalised), y_metres, well_id, district,
            last_input_date, target_date, split.

Run from the project root:  .venv/Scripts/python.exe -B -I preprocessing/c3_windows.py
"""

import hashlib
import json
import os
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{ROOT}/preprocessing/c2_normalised.csv"
SCALER = f"{ROOT}/preprocessing/c2_scaler.json"
SPLIT = f"{ROOT}/preprocessing/c4_split_dates.json"
CGWB = f"{ROOT}/data/cgwb_thanjavur_thiruvarur_nagapattinam_2013_2025.csv"
OUT_MD = f"{ROOT}/preprocessing/c3_report.md"
LWIN = 8
FEATURES = ["rainfall_180d_avg_mm", "tmax_90d_avg_c", "tmin_90d_avg_c",
            "humidity_90d_avg_pct", "et_90d_sum_mm", "depth_m"]
ZF = ["z_" + c for c in FEATURES]
SPLITS = ["train", "val", "test"]

L = []


def say(s=""):
    print(s)
    L.append(s)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def save_npz(path, arrays):
    """Deterministic .npz (np.savez stamps the current time into the zip, so files would differ per run)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            with zf.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.ascontiguousarray(arrays[name]), allow_pickle=False)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    with open(SPLIT) as f:
        cut = json.load(f)
    cut_train, cut_val = pd.Timestamp(cut["train_cutoff"]), pd.Timestamp(cut["val_cutoff"])

    raw = pd.read_csv(SRC, keep_default_na=False)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["well_id", "date"], kind="stable").reset_index(drop=True)
    for c in ["gap_from_prev_quarters", "gap_to_next_quarters", "next_depth", "z_next_depth"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["next_date"] = pd.to_datetime(raw["next_date"].replace("", np.nan))
    raw["has_target"] = raw["has_target"].astype(bool)
    district = (pd.read_csv(CGWB, dtype=str, keep_default_na=False).drop_duplicates("station_name")
                .set_index("station_name")["district_name"])
    raw["district"] = raw["well_id"].map(district)
    assert raw["district"].notna().all() and len(raw) == 1211

    # ---- scaler check: the C2 statistics use only readings on or before the train cutoff
    with open(SCALER) as f:
        sc = json.load(f)
    fitm = raw["date"] <= cut_train
    assert sc["n_readings_fit"] == int(fitm.sum())
    for c in FEATURES:
        assert np.isclose(sc["features"][c]["mean"], raw.loc[fitm, c].mean())
        assert np.isclose(sc["features"][c]["std"], raw.loc[fitm, c].std(ddof=0))
    # and the normalised columns really are (raw - train mean) / train std
    for c in FEATURES:
        s = sc["features"][c]
        assert np.allclose(raw["z_" + c], (raw[c] - s["mean"]) / s["std_used"], atol=1e-9)

    # ---- build windows
    recs = {"gap": [], "strict": []}
    loose = []
    for well, d in raw.groupby("well_id", sort=False):
        d = d.reset_index(drop=True)
        assert d["date"].is_monotonic_increasing
        Xall = d[ZF].values
        gprev = d["gap_from_prev_quarters"].values
        for k in range(LWIN - 1, len(d)):
            if not d.loc[k, "has_target"]:
                continue
            sl = slice(k - LWIN + 1, k + 1)
            rec = dict(X=Xall[sl], G=gprev[sl], tgap=int(d.loc[k, "gap_to_next_quarters"]),
                       y=float(d.loc[k, "z_next_depth"]), ym=float(d.loc[k, "next_depth"]),
                       well=well, dist=d.loc[k, "district"], last=d.loc[k, "date"],
                       first=d.loc[k - LWIN + 1, "date"], tdate=d.loc[k, "next_date"],
                       last_raw_depth=float(d.loc[k, "depth_m"]))
            recs["gap"].append(rec)
            if (gprev[k - LWIN + 2:k + 1] == 1).all():       # the 7 links inside the window
                loose.append(well)                           # kept for the report only (target gap not required)
                if rec["tgap"] == 1:                         # STRICT also needs the target in the very next quarter
                    recs["strict"].append(rec)

    out = {}
    for var in ("gap", "strict"):
        r = recs[var]
        tdate = pd.to_datetime([x["tdate"] for x in r])
        split = np.where(tdate <= cut_train, "train", np.where(tdate <= cut_val, "val", "test"))
        arr = dict(
            X=np.stack([x["X"] for x in r]).astype(np.float64),
            G=np.stack([x["G"] for x in r]).astype(np.float64),
            target_gap=np.array([x["tgap"] for x in r], dtype=np.int64),
            y=np.array([x["y"] for x in r], dtype=np.float64),
            y_metres=np.array([x["ym"] for x in r], dtype=np.float64),
            well_id=np.array([x["well"] for x in r], dtype=str),
            district=np.array([x["dist"] for x in r], dtype=str),
            last_input_date=np.array([x["last"].strftime("%Y-%m-%d") for x in r], dtype=str),
            target_date=np.array([x["tdate"].strftime("%Y-%m-%d") for x in r], dtype=str),
            split=split.astype(str))
        extra = dict(first=pd.to_datetime([x["first"] for x in r]), last=pd.to_datetime([x["last"] for x in r]),
                     tdate=tdate, last_raw_depth=np.array([x["last_raw_depth"] for x in r]))
        out[var] = (arr, extra)
        save_npz(f"{ROOT}/preprocessing/c3_windows_{var}.npz", arr)

    # ---- leakage / structure checks
    checks = []
    for var, (a, e) in out.items():
        N = len(a["y"])
        assert a["X"].shape == (N, LWIN, 6) and a["G"].shape == (N, LWIN)
        assert (e["tdate"] > e["last"]).all()
        checks.append(f"{var.upper()}: target_date > last_input_date for all {N} windows")
        tr, va, te = (a["split"] == s for s in SPLITS)
        assert (e["tdate"][tr] <= cut_train).all() and (e["tdate"][va | te] > cut_train).all()
        assert (e["tdate"][va] <= cut_val).all() and (e["tdate"][te] > cut_val).all()
        checks.append(f"{var.upper()}: train targets <= {cut['train_cutoff']}, val/test targets after it, val <= {cut['val_cutoff']} < test")
        # normalised target = (metres - train depth mean)/train depth std
        s = sc["features"]["depth_m"]
        assert np.allclose(a["y"], (a["y_metres"] - s["mean"]) / s["std_used"], atol=1e-9)
        # time order inside each well, and wells stay contiguous (never mixed, never shuffled)
        ids = a["well_id"]
        change = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        assert len(set(ids)) == len(change) + 1, "a well appears in two separate blocks"
        for blk in np.split(np.arange(N), change):
            assert np.all(np.diff(e["last"][blk].values.astype("datetime64[D]").astype(int)) > 0)
        checks.append(f"{var.upper()}: wells are contiguous blocks, windows in strictly increasing time inside each well (no mixing, no shuffling)")
        # each window's 8 rows really belong to its well and end at last_input_date (rebuilt independently)
        by = {w: d.reset_index(drop=True) for w, d in raw.groupby("well_id")}
        for i in range(N):
            d = by[ids[i]]
            k = int(np.flatnonzero(d["date"].values == e["last"][i].to_datetime64())[0])
            assert k >= LWIN - 1
            assert np.array_equal(d[ZF].values[k - LWIN + 1:k + 1], a["X"][i])
            assert d.loc[k, "next_depth"] == a["y_metres"][i]
        checks.append(f"{var.upper()}: all {N} X blocks and targets re-read independently from their own well rows: equal")
        # no window spans a gap that is not recorded: G agrees with the quarter gaps in the table
        assert np.isfinite(a["G"][:, 1:]).all()
        if var == "strict":
            assert (a["G"][:, 1:] == 1).all()
    a_g, a_s = out["gap"][0], out["strict"][0]
    # STRICT is a subset of GAP
    keys_g = set(zip(a_g["well_id"], a_g["last_input_date"]))
    assert all((w, t) in keys_g for w, t in zip(a_s["well_id"], a_s["last_input_date"]))
    checks.append("STRICT windows are a subset of GAP windows")
    checks.append(f"scaler (C2) statistics recomputed from readings dated <= {cut['train_cutoff']} ({int(fitm.sum())} readings): equal to c2_scaler.json")

    # ------------------------------------------------------------ report
    say("# C3 report: sliding windows (L = 8)")
    say()
    say(f"Input `preprocessing/c2_normalised.csv` (read only). A window = 8 readings of one well ending at reading k; target = depth of reading k+1 of the same well. Split by target date with the C4 cutoffs (train <= {cut['train_cutoff']}, val <= {cut['val_cutoff']}, test after). X = the 6 normalised features; G = gap in quarters before each position; shapes below.")
    say()
    say("## 1. Counts against the earlier hand counts")
    say()
    n_g, n_s = len(a_g["y"]), len(a_s["y"])
    w_g, w_s = len(set(a_g["well_id"])), len(set(a_s["well_id"]))
    assert (a_s["target_gap"] == 1).all()
    n_l, w_l = len(loose), len(set(loose))
    rows = [["GAP", n_g, w_g, "about 840 windows / 42 wells", "matches" if (n_g, w_g) == (840, 42) else "DIFFERS"],
            ["STRICT (8 consecutive quarters + target in the next quarter)", n_s, w_s, "about 157 windows / 23 wells", "matches" if (n_s, w_s) == (157, 23) else "DIFFERS"],
            ["for reference: 8 consecutive quarters, target gap unconstrained (not saved)", n_l, w_l, "-", "-"]]
    for line in md_table(["variant", "windows", "wells", "earlier hand count", "result"], rows):
        say(line)
    say()
    if (n_g, w_g) == (840, 42):
        say("- GAP reproduces the earlier count exactly.")
    if (n_s, w_s) == (157, 23):
        say(f"- STRICT reproduces the earlier count exactly once the target must be in the very next quarter (decision 2026-10-18). The earlier hand count had that rule; the first version of this script did not and gave {n_l} windows from {w_l} wells, so the {n_l - n_s} windows whose target is 2 or more quarters after the last input were removed.")
    else:
        say(f"- STRICT does NOT reproduce 157 / 23 ({n_s} windows / {w_s} wells); cause not identified by this script.")
    say()
    say(f"Array shapes: GAP X {a_g['X'].shape}, G {a_g['G'].shape}; STRICT X {a_s['X'].shape}, G {a_s['G'].shape}. Saved: `c3_windows_gap.npz`, `c3_windows_strict.npz` (arrays: X, G, target_gap, y, y_metres, well_id, district, last_input_date, target_date, split; no pickle).")
    nan0 = {v: int(np.isnan(out[v][0]["G"][:, 0]).sum()) for v in out}
    say(f"- `G[:, 0]` is NaN when the window starts at the first reading of its well (nothing before it): GAP {nan0['gap']} windows, STRICT {nan0['strict']} windows. `G[:, 1:]` is always defined. Left as NaN, not filled.")
    say()
    say("## 2. Windows and wells per split")
    say()
    rows = []
    for var, (a, e) in out.items():
        for s in SPLITS:
            m = a["split"] == s
            rows.append([var.upper(), s, int(m.sum()), len(set(a["well_id"][m])), f"{e['tdate'][m].min():%Y-%m-%d}" if m.any() else "-", f"{e['tdate'][m].max():%Y-%m-%d}" if m.any() else "-"])
    for line in md_table(["variant", "split", "windows", "wells", "first target", "last target"], rows):
        say(line)
    say()
    say("## 3. Windows per district")
    say()
    rows = []
    for var, (a, e) in out.items():
        for s in SPLITS + ["all"]:
            m = np.ones(len(a["y"]), bool) if s == "all" else a["split"] == s
            r = [var.upper(), s]
            for dname in ["Thanjavur", "Nagapattinam", "Thiruvarur"]:
                mm = m & (a["district"] == dname)
                r.append(f"{int(mm.sum())} / {len(set(a['well_id'][mm]))} wells")
            rows.append(r)
    for line in md_table(["variant", "split", "Thanjavur", "Nagapattinam", "Thiruvarur"], rows):
        say(line)
    say()
    say("- A STRICT split with no window in a district (see the table) means that district contributes nothing to training in that variant.")
    say()
    say("## 4. Gap from the last input to the target (quarters)")
    say()
    allg = sorted(set(a_g["target_gap"]) | set(a_s["target_gap"]))
    rows = []
    for gq in allg:
        rows.append([gq] + [f"{int((out[v][0]['target_gap'] == gq).sum())} ({100 * (out[v][0]['target_gap'] == gq).mean():.1f}%)" for v in ("gap", "strict")])
    for line in md_table(["target gap", "GAP windows", "STRICT windows"], rows):
        say(line)
    say()
    say("## 5. Span of an input window (first to last input reading, days)")
    say()
    rows = []
    for var, (a, e) in out.items():
        span = (e["last"] - e["first"]).days.values
        rows.append([var.upper(), int(np.median(span)), int(span.min()), int(span.max()), f"{span.mean():.0f}"])
    for line in md_table(["variant", "median", "min", "max", "mean"], rows):
        say(line)
    say()
    say("A strict 8-quarter window spans about 7 quarters (about 640 days); GAP windows can span many years where readings are sparse.")
    say()
    say("## 6. Persistence baseline (prediction = depth_m at the last input reading), MAE in metres")
    say()
    say("Descriptive baseline, not a model. Error = |depth_m(last input) - next depth|, on the raw metre values.")
    say()
    rows = []
    for var, (a, e) in out.items():
        for s in SPLITS:
            m = a["split"] == s
            err = np.abs(e["last_raw_depth"][m] - a["y_metres"][m])
            rows.append([var.upper(), s, int(m.sum()), f"{err.mean():.3f}" if m.any() else "-", f"{np.sqrt((err ** 2).mean()):.3f}" if m.any() else "-"])
    for line in md_table(["variant", "split", "windows", "MAE (m)", "RMSE (m)"], rows):
        say(line)
    say()
    neg = {v: int((out[v][0]["y_metres"] < 0).sum()) for v in out}
    negx = {v: int((out[v][0]["X"][:, :, 5] * sc["features"]["depth_m"]["std_used"] + sc["features"]["depth_m"]["mean"] < 0).any(axis=1).sum()) for v in out}
    say("## 7. Negative Allankottai depths (kept as they are)")
    say()
    say(f"- Windows whose target is negative: GAP {neg['gap']}, STRICT {neg['strict']}. Windows with a negative depth among the 8 inputs: GAP {negx['gap']}, STRICT {negx['strict']}.")
    say()
    say("## 8. Checks (asserted in code)")
    say()
    for c in checks:
        say(f"- {c}: passed.")
    say("- Re-running gives identical files: the npz files are written with a fixed zip timestamp and the CSV/JSON/MD outputs contain no timestamps. File fingerprints (first 16 hex of SHA-256) for comparison between runs: "
        f"gap `{sha(f'{ROOT}/preprocessing/c3_windows_gap.npz')}`, strict `{sha(f'{ROOT}/preprocessing/c3_windows_strict.npz')}`.")
    say()
    say("## 9. Files")
    say()
    say("- `preprocessing/c3_windows.py`, `c3_windows_gap.npz`, `c3_windows_strict.npz`, `c3_report.md`.")
    with open(OUT_MD, "w", newline="\n") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
