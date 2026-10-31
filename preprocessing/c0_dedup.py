"""
Phase C0: remove same-quarter duplicate readings. Nothing else.

Rule (locked): quarter = calendar quarter of the date (Q1 = Jan-Mar ... Q4 = Oct-Dec).
For each well_id and quarter keep the row with the latest date. No averaging, no
interpolation, no change to any value. Dates are used exactly as they are in the file.

Input  (read only): data/merged_final_2013_2025.csv
        data/cgwb_thanjavur_thiruvarur_nagapattinam_2013_2025.csv  (only for well -> district)
Output: preprocessing/c0_dedup.csv, c0_removed_duplicates.csv, c0_report.md

Run from the project root:  .venv/Scripts/python.exe -B -I preprocessing/c0_dedup.py
"""

import os
from decimal import Decimal

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{ROOT}/data/merged_final_2013_2025.csv"
CGWB = f"{ROOT}/data/cgwb_thanjavur_thiruvarur_nagapattinam_2013_2025.csv"
OUT = f"{ROOT}/preprocessing"
COLUMNS = ["well_id", "date", "rainfall_180d_avg_mm", "tmax_90d_avg_c", "tmin_90d_avg_c",
           "humidity_90d_avg_pct", "et_90d_sum_mm", "depth_to_water_level_m"]
EXPECTED_BEFORE, EXPECTED_AFTER = 1232, 1211

L = []                                   # report lines


def say(s=""):
    print(s)
    L.append(s)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


# ---------------------------------------------------------------- load (as text: values stay untouched)
with open(SRC, "rb") as f:
    eol = "\r\n" if b"\r\n" in f.read(4000) else "\n"
raw = pd.read_csv(SRC, dtype=str, keep_default_na=False)
assert list(raw.columns) == COLUMNS, raw.columns.tolist()
raw["_row"] = np.arange(len(raw))                       # original file order
raw["_date"] = pd.to_datetime(raw["date"], format="%Y-%m-%d")
raw["quarter"] = raw["_date"].dt.to_period("Q").astype(str)      # e.g. 2019Q1
raw["_depth"] = raw["depth_to_water_level_m"].astype(float)

n_before = len(raw)
wells_before = raw["well_id"].nunique()
assert (raw[COLUMNS] != "").all().all(), "empty cell in the input"

# ---------------------------------------------------------------- same date twice for one well?
same_date = raw[raw.duplicated(["well_id", "date"], keep=False)]
n_same_date_rows = len(same_date)

# ---------------------------------------------------------------- C0
s = raw.sort_values(["well_id", "_date", "_row"], kind="stable").reset_index(drop=True)
keep_mask = ~s.duplicated(["well_id", "quarter"], keep="last")   # ascending date -> last = latest
kept = s[keep_mask].copy()
removed = s[~keep_mask].copy()

pairs = removed.merge(kept, on=["well_id", "quarter"], suffixes=("_rem", "_kept"))
pairs["depth_difference"] = [format(Decimal(k) - Decimal(r), "f") for k, r in
                             zip(pairs["depth_to_water_level_m_kept"], pairs["depth_to_water_level_m_rem"])]
pairs["abs_diff"] = (pairs["_depth_kept"] - pairs["_depth_rem"]).abs()
pairs = pairs.sort_values(["well_id", "_date_rem"]).reset_index(drop=True)

out_removed = pd.DataFrame({
    "well_id": pairs["well_id"], "quarter": pairs["quarter"],
    "removed_date": pairs["date_rem"], "removed_depth": pairs["depth_to_water_level_m_rem"],
    "kept_date": pairs["date_kept"], "kept_depth": pairs["depth_to_water_level_m_kept"],
    "depth_difference": pairs["depth_difference"],
})
out_removed.to_csv(f"{OUT}/c0_removed_duplicates.csv", index=False, lineterminator=eol)

dedup = kept.sort_values(["well_id", "_date"], kind="stable").reset_index(drop=True)
dedup[COLUMNS + ["quarter"]].to_csv(f"{OUT}/c0_dedup.csv", index=False, lineterminator=eol)

# ---------------------------------------------------------------- checks (asserts + printed)
n_after = len(dedup)
n_removed = n_before - n_after
assert len(removed) == n_removed == len(pairs)
assert dedup["well_id"].nunique() == wells_before == 53
assert set(dedup["well_id"]) == set(raw["well_id"]), "a well lost all its readings"
assert list(dedup[COLUMNS].columns) == COLUMNS
assert (dedup[COLUMNS] != "").all().all()
assert not dedup.duplicated(["well_id", "quarter"]).any(), "more than one row per (well, quarter)"
# every kept row is an unmodified original row
chk = dedup[COLUMNS].merge(raw[COLUMNS], on=COLUMNS, how="left", indicator=True)
assert (chk["_merge"] == "both").all()
# every removed row really is an original row that lost to a later date in the same quarter
assert (pairs["_date_rem"] <= pairs["_date_kept"]).all()

# ---------------------------------------------------------------- report
say("# C0 report: same-quarter deduplication")
say()
say("Rule: for each well_id and calendar quarter keep the row with the latest date. No averaging, no interpolation, "
    "dates and values untouched. Source: `data/merged_final_2013_2025.csv` (not modified).")
say()
say("## 1. Counts and checks")
say()
ok_before = n_before == EXPECTED_BEFORE
ok_after = n_after == EXPECTED_AFTER
say(f"- Rows before: **{n_before}** (expected {EXPECTED_BEFORE}: {'as expected' if ok_before else 'DIFFERENT'}).")
say(f"- Rows after: **{n_after}** (expected {EXPECTED_AFTER}: {'as expected' if ok_after else 'DIFFERENT'}). "
    f"Removed: **{n_removed}** (expected 21: {'as expected' if n_removed == 21 else 'DIFFERENT'}).")
say(f"- Wells: {wells_before} before, {dedup['well_id'].nunique()} after. No well lost all its readings: passed.")
say(f"- Columns unchanged (the 8 original columns, plus `quarter` in `c0_dedup.csv`): passed.")
say(f"- Missing values after: {int((dedup[COLUMNS] == '').sum().sum())}: passed.")
say(f"- Exactly one row per (well_id, quarter) afterwards: passed.")
say(f"- Every kept row is identical, character for character, to an original row: passed.")
grp_sizes = s.groupby(["well_id", "quarter"]).size()
say(f"- Readings per (well, quarter) before: {int((grp_sizes == 1).sum())} slots with 1 reading, "
    f"{int((grp_sizes == 2).sum())} with 2, {int((grp_sizes >= 3).sum())} with 3 or more.")
say(f"- Same well and same date appearing twice: **{n_same_date_rows}** rows. "
    + ("None, so no tie had to be resolved." if n_same_date_rows == 0 else
       "Ties were resolved by keeping the later row in file order."))
say()

say("## 2. The removed readings")
say()
say("`depth_difference` = kept_depth - removed_depth (positive = the kept, later reading is deeper).")
say()
rows = [[r.well_id, r.quarter, r.removed_date, r.removed_depth, r.kept_date, r.kept_depth, r.depth_difference]
        for r in out_removed.itertuples()]
for line in md_table(["well_id", "quarter", "removed_date", "removed_depth", "kept_date", "kept_depth",
                      "depth_difference"], rows):
    say(line)
say()
imax = int(pairs["abs_diff"].idxmax())
gap_days = (pairs["_date_kept"] - pairs["_date_rem"]).dt.days
say(f"- Absolute depth difference between removed and kept reading: median **{pairs['abs_diff'].median():.3f} m**, "
    f"mean {pairs['abs_diff'].mean():.3f} m, maximum **{pairs['abs_diff'].max():.3f} m** "
    f"({pairs.loc[imax, 'well_id']}, {pairs.loc[imax, 'quarter']}: {pairs.loc[imax, 'date_rem']} depth "
    f"{pairs.loc[imax, 'depth_to_water_level_m_rem']} vs kept {pairs.loc[imax, 'date_kept']} depth "
    f"{pairs.loc[imax, 'depth_to_water_level_m_kept']}).")
say(f"- Days between removed and kept reading: median {gap_days.median():.0f}, minimum {gap_days.min()}, "
    f"maximum {gap_days.max()}.")
say(f"- Pairs where the two readings are exactly equal in depth: {int((pairs['abs_diff'] == 0).sum())}.")
say()

say("## 3. Specific cases")
say()
neg = raw[raw["_depth"] < 0].sort_values(["well_id", "_date"])
say(f"- Negative depths in the input: {len(neg)}. For each, whether its quarter holds a duplicate pair:")
kept_keys = set(zip(dedup["well_id"], dedup["date"]))
for r in neg.itertuples():
    in_dup = grp_sizes[(r.well_id, r.quarter)] > 1
    was_kept = (r.well_id, r.date) in kept_keys
    say(f"  - {r.well_id} {r.date} depth {r.depth_to_water_level_m} ({r.quarter}): "
        + ("**in a duplicate pair**" if in_dup else "not in a duplicate pair")
        + ("; kept in the output." if was_kept else "; REMOVED as the earlier reading of its pair."))
say("  Negative values were not changed.")
va = raw[(raw["well_id"] == "Vallam") & (raw["date"] == "2023-01-01")]
if len(va):
    r = va.iloc[0]
    others = raw[(raw["well_id"] == "Vallam") & (raw["quarter"] == r["quarter"])]
    say(f"- Vallam 2023-01-01 (portal date offset): its calendar quarter is **{r['quarter']}** using the date as written "
        f"(a 1-day shift would put it in 2022Q4, not applied). Readings of Vallam in {r['quarter']}: {len(others)} "
        + ("(" + ", ".join(f"{d} depth {x}" for d, x in zip(others['date'], others['depth_to_water_level_m'])) + ")")
        + (". Not in a duplicate pair." if len(others) == 1 else ". It IS in a duplicate pair."))
else:
    say("- Vallam 2023-01-01: not found in the input.")
ke = raw[(raw["well_id"] == "Keelvalur") & (raw["quarter"] == "2019Q1")].sort_values("_date")
kk = dedup[(dedup["well_id"] == "Keelvalur") & (dedup["quarter"] == "2019Q1")]
ke_txt = ", ".join(f"{d} depth {x}" for d, x in zip(ke["date"], ke["depth_to_water_level_m"]))
expected_ke = (list(ke["date"]) == ["2019-01-28", "2019-01-29"] and list(ke["depth_to_water_level_m"]) == ["3.15", "3.21"]
               and list(kk["depth_to_water_level_m"]) == ["3.21"])
say(f"- Keelvalur 2019Q1: {ke_txt}; kept: {', '.join(kk['date'] + ' depth ' + kk['depth_to_water_level_m'])}. "
    + ("Matches the expected example (keep 3.21)." if expected_ke else "DIFFERS from the expected example."))
say()

say("## 4. After deduplication")
say()
per_well = dedup.groupby("well_id").size()
say(f"- Readings per well: min {int(per_well.min())}, median {per_well.median():.0f}, max {int(per_well.max())}.")
district = (pd.read_csv(CGWB, dtype=str, keep_default_na=False).drop_duplicates("station_name")
            .set_index("station_name")["district_name"])
assert set(dedup["well_id"]) <= set(district.index)
dd = dedup["well_id"].map(district)
per_dist = dd.value_counts()
wells_per_dist = dedup.drop_duplicates("well_id")["well_id"].map(district).value_counts()
say("- Readings per district: " + ", ".join(f"{d} {int(per_dist[d])} ({int(wells_per_dist[d])} wells)"
                                            for d in ["Thanjavur", "Nagapattinam", "Thiruvarur"]) + ".")
say(f"- Occupied well-quarter slots: **{len(dedup.drop_duplicates(['well_id', 'quarter']))}** (= rows after dedup).")
say(f"- Distinct quarters covered: **{dedup['quarter'].nunique()}** "
    f"({dedup['quarter'].min()} to {dedup['quarter'].max()}).")
say(f"- Wells with 9 or more readings: {int((per_well >= 9).sum())}; with 17 or more: {int((per_well >= 17).sum())}.")
say()
say("## 5. Files")
say()
say("- `preprocessing/c0_dedup.csv`: the deduplicated table (8 original columns + `quarter`), sorted by well_id then date.")
say("- `preprocessing/c0_removed_duplicates.csv`: one row per removed reading.")
say("- `preprocessing/c0_dedup.py`: this script. `data/` was not touched.")

with open(f"{OUT}/c0_report.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(L) + "\n")
