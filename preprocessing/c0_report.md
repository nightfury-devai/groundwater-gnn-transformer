# C0 report: same-quarter deduplication

Rule: for each well_id and calendar quarter keep the row with the latest date. No averaging, no interpolation, dates and values untouched. Source: `data/merged_final_2013_2025.csv` (not modified).

## 1. Counts and checks

- Rows before: **1232** (expected 1232: as expected).
- Rows after: **1211** (expected 1211: as expected). Removed: **21** (expected 21: as expected).
- Wells: 53 before, 53 after. No well lost all its readings: passed.
- Columns unchanged (the 8 original columns, plus `quarter` in `c0_dedup.csv`): passed.
- Missing values after: 0: passed.
- Exactly one row per (well_id, quarter) afterwards: passed.
- Every kept row is identical, character for character, to an original row: passed.
- Readings per (well, quarter) before: 1190 slots with 1 reading, 21 with 2, 0 with 3 or more.
- Same well and same date appearing twice: **0** rows. None, so no tie had to be resolved.

## 2. The removed readings

`depth_difference` = kept_depth - removed_depth (positive = the kept, later reading is deeper).

| well_id | quarter | removed_date | removed_depth | kept_date | kept_depth | depth_difference |
|---|---|---|---|---|---|---|
| Keelvalur | 2019Q1 | 2019-01-28 | 3.15 | 2019-01-29 | 3.21 | 0.06 |
| Lower Anaicut | 2013Q1 | 2013-01-01 | 1.63 | 2013-01-04 | 2.13 | 0.50 |
| Mayavaram | 2013Q1 | 2013-01-01 | 0.91 | 2013-01-04 | 2.26 | 1.35 |
| Mayavaram | 2019Q1 | 2019-01-28 | 2.5 | 2019-01-29 | 2.76 | 0.26 |
| Melvastrachavdi | 2013Q1 | 2013-01-01 | 9.78 | 2013-01-04 | 19.13 | 9.35 |
| Nagapattinam | 2013Q1 | 2013-01-01 | 2.37 | 2013-01-04 | 4.19 | 1.82 |
| Nagore | 2019Q1 | 2019-01-28 | 2.95 | 2019-01-29 | 2.9 | -0.05 |
| Paramasivapuram II | 2019Q1 | 2019-01-28 | 17.0 | 2019-01-29 | 16.85 | -0.15 |
| Paramasivapuram III | 2013Q1 | 2013-01-01 | 6.3 | 2013-01-04 | 7.28 | 0.98 |
| Paramasivapuram III | 2019Q1 | 2019-01-28 | 10.35 | 2019-01-29 | 10.2 | -0.15 |
| Peralam | 2013Q1 | 2013-01-01 | 2.9 | 2013-01-04 | 1.17 | -1.73 |
| Puduchataram | 2013Q1 | 2013-01-01 | 1.63 | 2013-01-04 | 2.45 | 0.82 |
| Sirgazhi1 | 2013Q1 | 2013-01-01 | 2.27 | 2013-01-04 | 3.25 | 0.98 |
| Thagattur | 2013Q1 | 2013-01-01 | 1.71 | 2013-01-04 | 1.75 | 0.04 |
| Thagattur | 2019Q1 | 2019-01-28 | 1.5 | 2019-01-29 | 2.06 | 0.56 |
| Thirukadiyur | 2019Q1 | 2019-01-28 | 2.78 | 2019-01-29 | 2.83 | 0.05 |
| Tranquebar | 2013Q1 | 2013-01-01 | 0.75 | 2013-01-04 | 1.94 | 1.19 |
| Valangiman | 2013Q1 | 2013-01-01 | 2.26 | 2013-01-04 | 2.46 | 0.20 |
| Vallam | 2023Q1 | 2023-01-01 | 0.54 | 2023-01-20 | 3.4499998 | 2.9099998 |
| Vedaranyam | 2013Q1 | 2013-01-01 | 2.66 | 2013-01-04 | 0.65 | -2.01 |
| Velanganni | 2019Q1 | 2019-01-28 | 2.55 | 2019-01-29 | 2.37 | -0.18 |

- Absolute depth difference between removed and kept reading: median **0.560 m**, mean 1.207 m, maximum **9.350 m** (Melvastrachavdi, 2013Q1: 2013-01-01 depth 9.78 vs kept 2013-01-04 depth 19.13).
- Days between removed and kept reading: median 3, minimum 1, maximum 19.
- Pairs where the two readings are exactly equal in depth: 0.

## 3. Specific cases

- Negative depths in the input: 3. For each, whether its quarter holds a duplicate pair:
  - Allankottai 2021-01-10 depth -0.74 (2021Q1): not in a duplicate pair; kept in the output.
  - Allankottai 2023-01-20 depth -0.77 (2023Q1): not in a duplicate pair; kept in the output.
  - Allankottai 2024-11-30 depth -0.9 (2024Q4): not in a duplicate pair; kept in the output.
  Negative values were not changed.
- Vallam 2023-01-01 (portal date offset): its calendar quarter is **2023Q1** using the date as written (a 1-day shift would put it in 2022Q4, not applied). Readings of Vallam in 2023Q1: 2 (2023-01-01 depth 0.54, 2023-01-20 depth 3.4499998). It IS in a duplicate pair.
- Keelvalur 2019Q1: 2019-01-28 depth 3.15, 2019-01-29 depth 3.21; kept: 2019-01-29 depth 3.21. Matches the expected example (keep 3.21).

## 4. After deduplication

- Readings per well: min 1, median 26, max 42.
- Readings per district: Thanjavur 548 (26 wells), Nagapattinam 456 (16 wells), Thiruvarur 207 (11 wells).
- Occupied well-quarter slots: **1211** (= rows after dedup).
- Distinct quarters covered: **43** (2013Q1 to 2024Q4).
- Wells with 9 or more readings: 42; with 17 or more: 33.

## 5. Files

- `preprocessing/c0_dedup.csv`: the deduplicated table (8 original columns + `quarter`), sorted by well_id then date.
- `preprocessing/c0_removed_duplicates.csv`: one row per removed reading.
- `preprocessing/c0_dedup.py`: this script. `data/` was not touched.
