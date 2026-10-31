"""
Phase C4: chronological 70/15/15 split. No shuffling.

A sample's date is the date of its TARGET (the next reading), because that is the moment the
forecast is checked. Two cutoff dates on quarter boundaries are chosen so the gap-tolerant (GAP)
window set from C3 splits about 70/15/15 by target date. The SAME cutoffs are applied to the
strict set, so both variants share an identical test period.

Window membership (same rule as C3): a sample is a reading k that has a target and is at least the
8th reading of its well (so 8 readings up to and including k exist).
  GAP    : no constraint on the gaps between the 8 readings.
  STRICT : the 7 links inside the 8 readings are all exactly 1 quarter AND the target is the very
           next quarter (gap_to_next = 1), i.e. 9 consecutive quarters (decision 2026-10-18).

Input  (read only): preprocessing/c1_features.csv, data/cgwb_*.csv (well -> district only)
Output: preprocessing/c4_split_dates.json, preprocessing/c4_report.md

Run from the project root:  .venv/Scripts/python.exe -B -I preprocessing/c4_split.py
"""

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{ROOT}/preprocessing/c1_features.csv"
CGWB = f"{ROOT}/data/cgwb_thanjavur_thiruvarur_nagapattinam_2013_2025.csv"
OUT_JSON = f"{ROOT}/preprocessing/c4_split_dates.json"
OUT_MD = f"{ROOT}/preprocessing/c4_report.md"
L_WIN = 8
TARGET_FRAC = (0.70, 0.15, 0.15)

L = []


def say(s=""):
    print(s)
    L.append(s)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


# ------------------------------------------------------------ samples (one per reading that can end a window)
df = pd.read_csv(SRC, keep_default_na=False)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["well_id", "date"], kind="stable").reset_index(drop=True)
g = df.groupby("well_id", sort=False)
df["pos"] = g.cumcount()                                    # 0-based position of the reading in its well
df["has_target"] = df["has_target"].astype(bool)
# the 7 links inside a window ending at k are gap_from_prev of readings k-6..k (all must be 1 for STRICT)
df["gap_from_prev_quarters"] = pd.to_numeric(df["gap_from_prev_quarters"], errors="coerce")  # empty -> NaN (first reading)
g = df.groupby("well_id", sort=False)
links_ok = g["gap_from_prev_quarters"].transform(lambda s: (s == 1).astype(int).rolling(L_WIN - 1).sum()) == L_WIN - 1
df["is_gap"] = df["has_target"] & (df["pos"] >= L_WIN - 1)
df["gap_to_next_quarters"] = pd.to_numeric(df["gap_to_next_quarters"], errors="coerce")
# STRICT (decision 2026-10-18): 8 consecutive quarters AND the target is the very next quarter (9 consecutive quarters)
df["is_strict"] = df["is_gap"] & links_ok & (df["gap_to_next_quarters"] == 1)
df["target_date"] = pd.to_datetime(df["next_date"].replace("", np.nan))

district = (pd.read_csv(CGWB, dtype=str, keep_default_na=False).drop_duplicates("station_name")
            .set_index("station_name")["district_name"])
df["district"] = df["well_id"].map(district)
assert df["district"].notna().all()

gap = df[df["is_gap"]].copy()
strict = df[df["is_strict"]].copy()
assert (gap["target_date"] > gap["date"]).all()
n_gap = len(gap)

# ------------------------------------------------------------ choose cutoffs on quarter ends
quarter_ends = pd.date_range("2013-03-31", "2025-12-31", freq="QE")
cum = np.array([(gap["target_date"] <= q).sum() for q in quarter_ends])
frac = cum / n_gap


def best(target):
    # closest cumulative fraction to the target; first (earliest) quarter wins ties
    return int(np.argmin(np.abs(frac - target)))


i1 = best(TARGET_FRAC[0])
i2 = best(TARGET_FRAC[0] + TARGET_FRAC[1])
assert i1 < i2, "cutoffs collapsed"
cut_train, cut_val = quarter_ends[i1], quarter_ends[i2]


def assign(t):
    return np.where(t <= cut_train, "train", np.where(t <= cut_val, "val", "test"))


gap["split"] = assign(gap["target_date"].values)
strict["split"] = assign(strict["target_date"].values)
splits = ["train", "val", "test"]

with open(OUT_JSON, "w", newline="\n") as f:
    json.dump({"train_cutoff": cut_train.strftime("%Y-%m-%d"), "val_cutoff": cut_val.strftime("%Y-%m-%d"),
               "rule": "split by TARGET date: train <= train_cutoff < val <= val_cutoff < test",
               "window_length": L_WIN, "quarter_boundaries": True,
               "gap_samples_total": int(n_gap), "strict_samples_total": int(len(strict))},
              f, indent=2, sort_keys=True)
    f.write("\n")

# ------------------------------------------------------------ checks
say("# C4 report: chronological 70/15/15 split")
say()
say("Samples are split by the date of their TARGET reading (`next_date`). No shuffling. Input: `preprocessing/c1_features.csv` (read only). The cutoffs below are used by C2 (scaler fit) and C3 (window split).")
say()
say("## 1. Cutoffs")
say()
say(f"- Train/val cutoff: **{cut_train:%Y-%m-%d}** (a train sample has target date on or before this).")
say(f"- Val/test cutoff: **{cut_val:%Y-%m-%d}** (a val sample has target date after the train cutoff and on or before this; a test sample is after it).")
say(f"- Both are quarter ends. They were chosen as the quarter ends whose cumulative share of the {n_gap} GAP samples (by target date) is closest to 70% and 85%.")
say(f"- Saved to `preprocessing/c4_split_dates.json`.")
say()
say("## 2. Samples per split")
say()
rows = []
for name, d in (("GAP (gap-tolerant)", gap), ("STRICT (8 consecutive quarters + next-quarter target)", strict)):
    n = len(d)
    for s in splits:
        k = int((d["split"] == s).sum())
        rows.append([name, s, k, f"{100 * k / n:.1f}%"])
    rows.append([name, "total", n, "100%"])
for line in md_table(["variant", "split", "samples", "share"], rows):
    say(line)
say()
say("- STRICT uses the GAP cutoffs unchanged, so its shares differ from 70/15/15 (all STRICT windows are a subset of the GAP ones; the test period is identical in both).")
say()
say("## 3. Wells and districts per split")
say()
rows = []
for name, d in (("GAP", gap), ("STRICT", strict)):
    for s in splits:
        x = d[d["split"] == s]
        dc = x.groupby("district")["well_id"].agg(["count", "nunique"])
        parts = ", ".join(f"{k} {int(v['count'])} samples / {int(v['nunique'])} wells" for k, v in dc.sort_index().iterrows())
        rows.append([name, s, x["well_id"].nunique(), x["district"].nunique(), parts if parts else "none"])
for line in md_table(["variant", "split", "wells", "districts", "per district"], rows):
    say(line)
say()
say("## 4. Target-date range per split")
say()
rows = []
for name, d in (("GAP", gap), ("STRICT", strict)):
    for s in splits:
        x = d[d["split"] == s]["target_date"]
        rows.append([name, s, f"{x.min():%Y-%m-%d}" if len(x) else "-", f"{x.max():%Y-%m-%d}" if len(x) else "-"])
for line in md_table(["variant", "split", "first target date", "last target date"], rows):
    say(line)
say()
say("## 5. Checks (asserted in code)")
say()
for d, name in ((gap, "GAP"), (strict, "STRICT")):
    assert (d.loc[d["split"] == "train", "target_date"] <= cut_train).all()
    assert (d.loc[d["split"] != "train", "target_date"] > cut_train).all()
    assert (d.loc[d["split"] == "val", "target_date"] <= cut_val).all()
    assert (d.loc[d["split"] == "test", "target_date"] > cut_val).all()
    # chronological: every train target precedes every val target precedes every test target
    assert d.loc[d["split"] == "train", "target_date"].max() < d.loc[d["split"] == "val", "target_date"].min()
    assert d.loc[d["split"] == "val", "target_date"].max() < d.loc[d["split"] == "test", "target_date"].min()
    say(f"- {name}: train targets <= train cutoff; val targets in (train cutoff, val cutoff]; test targets > val cutoff; train < val < test in time: passed.")
assert set(strict.index) <= set(gap.index)
say("- Every STRICT sample is also a GAP sample: passed.")
assert cut_train.is_quarter_end and cut_val.is_quarter_end
say("- Cutoffs are quarter ends: passed.")
say("- Nothing was shuffled; the order is well, then date (stable sort).")
say()
say("## 6. Files")
say()
say("- `preprocessing/c4_split.py`, `preprocessing/c4_split_dates.json`, `preprocessing/c4_report.md`.")

with open(OUT_MD, "w", newline="\n") as f:
    f.write("\n".join(L) + "\n")
