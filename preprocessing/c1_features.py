"""
Phase C1: per-reading features and the target. Nothing else (no normalising, no windows, no split).

Decision 2026-10-14: the depth feature of a row is the depth measured AT THAT READING (depth_m).
The target of a row is the depth of the NEXT reading of the same well (next_depth), so every
row is a one-reading-ahead forecast.

Input  (read only): preprocessing/c0_dedup.csv   (1,211 rows, one per well and quarter)
        data/cgwb_thanjavur_thiruvarur_nagapattinam_2013_2025.csv  (only well -> district)
Output: preprocessing/c1_features.csv, preprocessing/c1_report.md

Nothing is changed, filled or interpolated: every value written is a value that is in the input.

Run from the project root:  .venv/Scripts/python.exe -B -I preprocessing/c1_features.py
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{ROOT}/preprocessing/c0_dedup.csv"
CGWB = f"{ROOT}/data/cgwb_thanjavur_thiruvarur_nagapattinam_2013_2025.csv"
OUT_CSV = f"{ROOT}/preprocessing/c1_features.csv"
OUT_MD = f"{ROOT}/preprocessing/c1_report.md"

FEATURES = ["rainfall_180d_avg_mm", "tmax_90d_avg_c", "tmin_90d_avg_c",
            "humidity_90d_avg_pct", "et_90d_sum_mm", "depth_m"]          # F = 6
OUT_COLUMNS = (["well_id", "date", "quarter"] + FEATURES +
               ["next_date", "next_quarter", "next_depth",
                "gap_to_next_quarters", "gap_from_prev_quarters", "has_target"])
EXPECTED_ROWS, EXPECTED_TARGETS = 1211, 1158

L = []


def say(s=""):
    print(s)
    L.append(s)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def quarter_index(q):
    """'2019Q1' -> whole-quarter counter, so differences are whole quarters."""
    return int(q[:4]) * 4 + int(q[5]) - 1


# ------------------------------------------------------------ load (as text, so no value can change)
with open(SRC, "rb") as f:
    eol = "\r\n" if b"\r\n" in f.read(4000) else "\n"
src = pd.read_csv(SRC, dtype=str, keep_default_na=False)
assert len(src) == EXPECTED_ROWS, len(src)
assert (src != "").all().all(), "empty cell in the input"
src["_date"] = pd.to_datetime(src["date"], format="%Y-%m-%d")
src["_qidx"] = src["quarter"].map(quarter_index)
# per well, ascending date (stable, so equal dates would keep file order; there are none after C0)
src = src.sort_values(["well_id", "_date"], kind="stable").reset_index(drop=True)
assert not src.duplicated(["well_id", "quarter"]).any()

# ------------------------------------------------------------ build the columns
g = src.groupby("well_id", sort=False)
out = pd.DataFrame({"well_id": src["well_id"], "date": src["date"], "quarter": src["quarter"]})
# the 6 features are straight copies of the SAME row's values
for c in FEATURES[:-1]:
    out[c] = src[c]
out["depth_m"] = src["depth_to_water_level_m"]
# values of the NEXT row of the same well (shift(-1) inside each well; the last row of a well gets NaN)
nxt_date = g["date"].shift(-1)
nxt_quarter = g["quarter"].shift(-1)
nxt_depth = g["depth_to_water_level_m"].shift(-1)
nxt_qidx = g["_qidx"].shift(-1)
prv_qidx = g["_qidx"].shift(1)
out["next_date"] = nxt_date.fillna("")
out["next_quarter"] = nxt_quarter.fillna("")
out["next_depth"] = nxt_depth.fillna("")
out["gap_to_next_quarters"] = (nxt_qidx - src["_qidx"]).map(lambda v: "" if pd.isna(v) else str(int(v)))
out["gap_from_prev_quarters"] = (src["_qidx"] - prv_qidx).map(lambda v: "" if pd.isna(v) else str(int(v)))
out["has_target"] = nxt_depth.notna()
out = out[OUT_COLUMNS]
out.to_csv(OUT_CSV, index=False, lineterminator=eol)

# ------------------------------------------------------------ checks
n_rows = len(out)
has = out["has_target"]
n_target = int(has.sum())
n_wells = out["well_id"].nunique()
assert n_rows == EXPECTED_ROWS
assert list(out.columns) == OUT_COLUMNS

# (1) next_date > date, and next_depth == the next row's depth_m, verified with an explicit loop
#     (independent of the shift() used to build the columns)
loop_ok = loop_checked = 0
last_rows = 0
for well, d in out.groupby("well_id", sort=False):
    d = d.reset_index(drop=True)
    assert d["date"].is_monotonic_increasing and d["date"].is_unique
    for i in range(len(d)):
        if i == len(d) - 1:
            assert d.loc[i, "next_date"] == "" and d.loc[i, "next_depth"] == "" and not d.loc[i, "has_target"]
            assert d.loc[i, "gap_to_next_quarters"] == ""
            last_rows += 1
        else:
            assert d.loc[i, "next_depth"] == d.loc[i + 1, "depth_m"], (well, i)
            assert d.loc[i, "next_date"] == d.loc[i + 1, "date"]
            assert d.loc[i, "next_quarter"] == d.loc[i + 1, "quarter"]
            assert pd.Timestamp(d.loc[i, "next_date"]) > pd.Timestamp(d.loc[i, "date"])
            assert d.loc[i, "has_target"]
            # the gap seen from this row and from the next row must agree
            assert d.loc[i, "gap_to_next_quarters"] == d.loc[i + 1, "gap_from_prev_quarters"]
            loop_ok += 1
        loop_checked += 1
assert loop_checked == n_rows and last_rows == n_wells
assert loop_ok == n_target
tg = out[has]
assert (pd.to_datetime(tg["next_date"]) > pd.to_datetime(tg["date"])).all()
assert tg["gap_to_next_quarters"].astype(int).min() >= 1
# (2) every feature value is character-for-character the value of the same row of the input
src_chk = src[["well_id", "date"] + FEATURES[:-1] + ["depth_to_water_level_m"]].rename(
    columns={"depth_to_water_level_m": "depth_m"})
m = out[["well_id", "date"] + FEATURES].merge(src_chk, on=["well_id", "date"], suffixes=("", "_src"), how="left")
for c in FEATURES:
    assert (m[c] == m[c + "_src"]).all(), c
# (3) none of the 6 features is one of the columns built from the next row
next_cols = {"next_date", "next_quarter", "next_depth", "gap_to_next_quarters"}
assert not (set(FEATURES) & next_cols)
assert out[FEATURES].map(lambda v: v != "").all().all()      # no missing feature value

# ------------------------------------------------------------ report
say("# C1 report: features and target")
say()
say("Decision 2026-10-14: the depth feature of a row is the depth measured at that reading (`depth_m`); the target is "
    "the depth of the next reading of the same well (`next_depth`). Source: `preprocessing/c0_dedup.csv` "
    "(read only). `data/` was not touched. Nothing was normalised, windowed or split (that is C2 to C4).")
say()
say("## 1. Columns of `preprocessing/c1_features.csv`")
say()
say("- Keys: `well_id`, `date`, `quarter`.")
say("- The 6 model features (F = 6): " + ", ".join(f"`{c}`" for c in FEATURES) +
    ". `depth_m` is a copy of `depth_to_water_level_m` at the same reading "
    "(the original column is not repeated).")
say("- From the next row of the same well: `next_date`, `next_quarter`, `next_depth` (the target). Empty for the "
    "last reading of a well.")
say("- `gap_to_next_quarters` (whole quarters from this quarter to `next_quarter`) and `gap_from_prev_quarters` "
    "(from the previous row's quarter to this quarter). Kept for later; NOT among the 6 features.")
say("- `has_target`: True if `next_depth` exists.")
say()
say("## 2. Checks")
say()
say(f"- Rows: **{n_rows}** (expected {EXPECTED_ROWS}: {'as expected' if n_rows == EXPECTED_ROWS else 'DIFFERENT'}). "
    f"Wells: {n_wells}.")
say(f"- Rows with a target: **{n_target}** (expected {EXPECTED_TARGETS} = {EXPECTED_ROWS} minus 53 last readings: "
    f"{'as expected' if n_target == EXPECTED_TARGETS else 'DIFFERENT'}). Rows without a target: {n_rows - n_target} "
    f"(= {last_rows} last readings, one per well).")
say(f"- `next_date` is later than `date` for all {n_target} rows with a target: passed.")
say(f"- `next_depth` equals the `depth_m` of the next row of the same well, `next_date` and `next_quarter` equal "
    f"that row's date and quarter: passed for all {loop_ok} rows (explicit per-well loop, independent of the "
    f"shift used to build the columns).")
say("- No feature uses a value from the next row. How this is guaranteed: (a) each of the 6 feature columns is "
    "a copy of the same row's value and is asserted equal, character for character, to that row in the input; "
    "(b) the only columns taken from the next row are `next_date`, `next_quarter`, `next_depth` and "
    "`gap_to_next_quarters`, and none of them is in the feature list (asserted); (c) the 5 weather features were "
    "computed in the merge from trailing windows that end on the reading's own date, so they contain no weather "
    "after that reading.")
say(f"- No missing value in any of the 6 features: passed. No value was changed, filled or interpolated.")
say()

say("## 3. Gaps in quarters")
say()
gt = tg["gap_to_next_quarters"].astype(int)
gf_all = out["gap_from_prev_quarters"]
gf = gf_all[gf_all != ""].astype(int)
say(f"Gap from this reading to the next one (`gap_to_next_quarters`), {len(gt)} rows with a target:")
say()
rows = [[int(k), int(v), f"{100 * v / len(gt):.1f}%"] for k, v in gt.value_counts().sort_index().items()]
for line in md_table(["quarters", "rows", "percent"], rows):
    say(line)
say()
say(f"- Median {gt.median():.0f}, mean {gt.mean():.2f}, maximum {int(gt.max())}. "
    f"Exactly 1 quarter: {100 * (gt == 1).mean():.1f}% of the rows; 4 quarters or more: {100 * (gt >= 4).mean():.1f}%.")
say()
say(f"Gap from the previous reading to this one (`gap_from_prev_quarters`), {len(gf)} rows with a previous reading "
    f"({n_rows - len(gf)} first readings have none):")
say()
rows = [[int(k), int(v), f"{100 * v / len(gf):.1f}%"] for k, v in gf.value_counts().sort_index().items()]
for line in md_table(["quarters", "rows", "percent"], rows):
    say(line)
say()
say(f"- The two distributions are identical in counts (each gap is seen once from each side): "
    f"{'yes' if gt.value_counts().sort_index().equals(gf.value_counts().sort_index()) else 'NO'}.")
say()

say("## 4. Rows per district")
say()
district = (pd.read_csv(CGWB, dtype=str, keep_default_na=False).drop_duplicates("station_name")
            .set_index("station_name")["district_name"])
assert set(out["well_id"]) <= set(district.index)
dist = out["well_id"].map(district)
rows = []
for d in ["Thanjavur", "Nagapattinam", "Thiruvarur"]:
    sel = dist == d
    rows.append([d, int(out.loc[sel, "well_id"].nunique()), int(sel.sum()), int((sel & has).sum())])
rows.append(["Total", n_wells, n_rows, n_target])
for line in md_table(["district", "wells", "readings", "rows with a target"], rows):
    say(line)
per_well_t = out[has].groupby("well_id").size()
say()
say(f"- Wells with at least one row with a target: {per_well_t.size} of {n_wells}. "
    f"Wells with a single reading (no target): {n_wells - per_well_t.size}.")
say(f"- Rows with a target per well: min {int(per_well_t.min())}, median {per_well_t.median():.0f}, "
    f"max {int(per_well_t.max())}.")
say()

say("## 5. Example: Keelvalur 2019")
say()
ke = out[(out["well_id"] == "Keelvalur") & (out["date"].str.startswith("2019"))]
rows = [[r.date, r.quarter, r.depth_m, r.next_date, r.next_quarter, r.next_depth, r.gap_to_next_quarters,
         r.gap_from_prev_quarters] for r in ke.itertuples()]
for line in md_table(["date", "quarter", "depth_m", "next_date", "next_quarter", "next_depth", "gap_to_next",
                      "gap_from_prev"], rows):
    say(line)
say()
exp = ke[(ke["date"] == "2019-05-29")]
ok = (len(exp) == 1 and exp.iloc[0]["depth_m"] == "4.26" and exp.iloc[0]["next_date"] == "2019-08-29"
      and exp.iloc[0]["next_depth"] == "3.76")
say("- Expected: 2019-05-29 depth 4.26 has next_depth 3.76 on 2019-08-29. "
    + ("Matches." if ok else "DIFFERS: see the table above."))
say()

say("## 6. Persistence level (descriptive only, all data, NOT a result)")
say()
x = tg["depth_m"].astype(float).to_numpy()
y = tg["next_depth"].astype(float).to_numpy()
r = float(np.corrcoef(x, y)[0, 1])
mae = float(np.mean(np.abs(y - x)))
say(f"Over the {n_target} rows with a target, comparing `depth_m` (this reading) with `next_depth` (next reading), "
    f"i.e. the 'next = last' persistence level:")
say()
say(f"- Pearson r = **{r:.3f}**.")
say(f"- Mean absolute error = **{mae:.3f} m** (mean `next_depth` {y.mean():.2f} m, standard deviation {y.std(ddof=0):.2f} m).")
say()
say("This uses all rows, with no train/test split and regardless of how many quarters lie between the two "
    "readings. It describes the data; it is not a model result and not a baseline score.")
say()
say("## 7. Files")
say()
say("- `preprocessing/c1_features.csv`: the table described in section 1 (1,211 rows, 15 columns).")
say("- `preprocessing/c1_features.py`: this script.")

with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(L) + "\n")
