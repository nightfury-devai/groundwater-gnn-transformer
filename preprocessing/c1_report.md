# C1 report: features and target

Decision 2026-10-14: the depth feature of a row is the depth measured at that reading (`depth_m`); the target is the depth of the next reading of the same well (`next_depth`). Source: `preprocessing/c0_dedup.csv` (read only). `data/` was not touched. Nothing was normalised, windowed or split (that is C2 to C4).

## 1. Columns of `preprocessing/c1_features.csv`

- Keys: `well_id`, `date`, `quarter`.
- The 6 model features (F = 6): `rainfall_180d_avg_mm`, `tmax_90d_avg_c`, `tmin_90d_avg_c`, `humidity_90d_avg_pct`, `et_90d_sum_mm`, `depth_m`. `depth_m` is a copy of `depth_to_water_level_m` at the same reading (the original column is not repeated).
- From the next row of the same well: `next_date`, `next_quarter`, `next_depth` (the target). Empty for the last reading of a well.
- `gap_to_next_quarters` (whole quarters from this quarter to `next_quarter`) and `gap_from_prev_quarters` (from the previous row's quarter to this quarter). Kept for later; NOT among the 6 features.
- `has_target`: True if `next_depth` exists.

## 2. Checks

- Rows: **1211** (expected 1211: as expected). Wells: 53.
- Rows with a target: **1158** (expected 1158 = 1211 minus 53 last readings: as expected). Rows without a target: 53 (= 53 last readings, one per well).
- `next_date` is later than `date` for all 1158 rows with a target: passed.
- `next_depth` equals the `depth_m` of the next row of the same well, `next_date` and `next_quarter` equal that row's date and quarter: passed for all 1158 rows (explicit per-well loop, independent of the shift used to build the columns).
- No feature uses a value from the next row. How this is guaranteed: (a) each of the 6 feature columns is a copy of the same row's value and is asserted equal, character for character, to that row in the input; (b) the only columns taken from the next row are `next_date`, `next_quarter`, `next_depth` and `gap_to_next_quarters`, and none of them is in the feature list (asserted); (c) the 5 weather features were computed in the merge from trailing windows that end on the reading's own date, so they contain no weather after that reading.
- No missing value in any of the 6 features: passed. No value was changed, filled or interpolated.

## 3. Gaps in quarters

Gap from this reading to the next one (`gap_to_next_quarters`), 1158 rows with a target:

| quarters | rows | percent |
|---|---|---|
| 1 | 839 | 72.5% |
| 2 | 181 | 15.6% |
| 3 | 86 | 7.4% |
| 4 | 15 | 1.3% |
| 5 | 6 | 0.5% |
| 6 | 4 | 0.3% |
| 7 | 4 | 0.3% |
| 8 | 6 | 0.5% |
| 9 | 11 | 0.9% |
| 10 | 2 | 0.2% |
| 11 | 3 | 0.3% |
| 21 | 1 | 0.1% |

- Median 1, mean 1.57, maximum 21. Exactly 1 quarter: 72.5% of the rows; 4 quarters or more: 4.5%.

Gap from the previous reading to this one (`gap_from_prev_quarters`), 1158 rows with a previous reading (53 first readings have none):

| quarters | rows | percent |
|---|---|---|
| 1 | 839 | 72.5% |
| 2 | 181 | 15.6% |
| 3 | 86 | 7.4% |
| 4 | 15 | 1.3% |
| 5 | 6 | 0.5% |
| 6 | 4 | 0.3% |
| 7 | 4 | 0.3% |
| 8 | 6 | 0.5% |
| 9 | 11 | 0.9% |
| 10 | 2 | 0.2% |
| 11 | 3 | 0.3% |
| 21 | 1 | 0.1% |

- The two distributions are identical in counts (each gap is seen once from each side): yes.

## 4. Rows per district

| district | wells | readings | rows with a target |
|---|---|---|---|
| Thanjavur | 26 | 548 | 522 |
| Nagapattinam | 16 | 456 | 440 |
| Thiruvarur | 11 | 207 | 196 |
| Total | 53 | 1211 | 1158 |

- Wells with at least one row with a target: 49 of 53. Wells with a single reading (no target): 4.
- Rows with a target per well: min 2, median 27, max 41.

## 5. Example: Keelvalur 2019

| date | quarter | depth_m | next_date | next_quarter | next_depth | gap_to_next | gap_from_prev |
|---|---|---|---|---|---|---|---|
| 2019-01-29 | 2019Q1 | 3.21 | 2019-05-29 | 2019Q2 | 4.26 | 1 | 1 |
| 2019-05-29 | 2019Q2 | 4.26 | 2019-08-29 | 2019Q3 | 3.76 | 1 | 1 |
| 2019-08-29 | 2019Q3 | 3.76 | 2019-11-29 | 2019Q4 | 1.86 | 1 | 1 |
| 2019-11-29 | 2019Q4 | 1.86 | 2020-01-09 | 2020Q1 | 2.91 | 1 | 1 |

- Expected: 2019-05-29 depth 4.26 has next_depth 3.76 on 2019-08-29. Matches.

## 6. Persistence level (descriptive only, all data, NOT a result)

Over the 1158 rows with a target, comparing `depth_m` (this reading) with `next_depth` (next reading), i.e. the 'next = last' persistence level:

- Pearson r = **0.821**.
- Mean absolute error = **1.674 m** (mean `next_depth` 4.69 m, standard deviation 4.05 m).

This uses all rows, with no train/test split and regardless of how many quarters lie between the two readings. It describes the data; it is not a model result and not a baseline score.

## 7. Files

- `preprocessing/c1_features.csv`: the table described in section 1 (1,211 rows, 15 columns).
- `preprocessing/c1_features.py`: this script.
