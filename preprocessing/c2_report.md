# C2 report: z-score normalisation (train-fitted)

Input `preprocessing/c1_features.csv` (read only, 1211 readings). Cutoffs from `preprocessing/c4_split_dates.json`: train 2022-09-30, val 2023-12-31.

## 1. Rule

- Mean and std come ONLY from readings dated on or before the train cutoff (2022-09-30): **952 of 1211 readings** (latest fitted reading date 2022-08-30). The 259 later readings were not used.
- Population std (ddof = 0). Guard: a std of 0 would be replaced by 1; no feature needed it.
- `depth_m` and `next_depth` share one pair of statistics (the `depth_m` ones). `z_next_depth` is empty where the target is empty (53 last readings).
- No value was filled, interpolated or removed. Saved: `preprocessing/c2_scaler.json` (with `min`, `max`, `n`) and `preprocessing/c2_normalised.csv` (all C1 columns, unchanged, plus 7 `z_` columns).
- Inverse: `depth_to_metres(z, scaler)` and `feature_to_raw(name, z, scaler)` in `c2_normalise.py`; checked here to reproduce all raw values to 1e-9.
- Negative depths kept as they are: 3 readings with depth_m < 0 in the data (1 of them inside the fit period and therefore in the train statistics).

## 2. Train-fitted statistics

| feature | n readings | mean | std | train min | train max |
|---|---|---|---|---|---|
| rainfall_180d_avg_mm | 952 | 3.1556 | 2.3026 | 0.0000 | 10.3293 |
| tmax_90d_avg_c | 952 | 33.6310 | 2.2225 | 30.0099 | 37.6777 |
| tmin_90d_avg_c | 952 | 24.9864 | 1.3172 | 22.0975 | 26.9957 |
| humidity_90d_avg_pct | 952 | 73.1053 | 8.4980 | 52.0804 | 88.0176 |
| et_90d_sum_mm | 952 | 53.2909 | 46.3172 | 0.0204 | 159.3755 |
| depth_m | 952 | 4.7514 | 3.9841 | -0.7400 | 20.8500 |

## 3. Same statistics if fitted on ALL data (for comparison only, NOT used)

| feature | mean (train) | mean (all) | diff | std (train) | std (all) | diff |
|---|---|---|---|---|---|---|
| rainfall_180d_avg_mm | 3.1556 | 3.2174 | +0.0619 | 2.3026 | 2.2200 | -0.0826 |
| tmax_90d_avg_c | 33.6310 | 33.6487 | +0.0177 | 2.2225 | 2.1892 | -0.0334 |
| tmin_90d_avg_c | 24.9864 | 24.9858 | -0.0006 | 1.3172 | 1.3083 | -0.0089 |
| humidity_90d_avg_pct | 73.1053 | 73.4698 | +0.3646 | 8.4980 | 8.2547 | -0.2433 |
| et_90d_sum_mm | 53.2909 | 56.4798 | +3.1890 | 46.3172 | 46.7522 | +0.4350 |
| depth_m | 4.7514 | 4.6821 | -0.0693 | 3.9841 | 4.0238 | +0.0397 |

- All-data fit uses 1211 readings, train fit 952. The difference is the leakage a global scaler would add.

## 4. Val/test values outside the train min-max range

Counted over the 1158 rows that have a target, each assigned to val or test by its TARGET date (rows with target date on or before the train cutoff are train). 'Outside' means below the train min or above the train max of that feature (fit-period readings). Expected for a chronological split (the climate and the water table move); reported, not corrected, nothing is clipped. `next_depth` uses the depth range.

| column | val (n=147) | test (n=112) |
|---|---|---|
| rainfall_180d_avg_mm | 0 of 147 (0.0%) | 0 of 112 (0.0%) |
| tmax_90d_avg_c | 0 of 147 (0.0%) | 0 of 112 (0.0%) |
| tmin_90d_avg_c | 2 of 147 (1.4%) | 0 of 112 (0.0%) |
| humidity_90d_avg_pct | 0 of 147 (0.0%) | 0 of 112 (0.0%) |
| et_90d_sum_mm | 0 of 147 (0.0%) | 0 of 112 (0.0%) |
| depth_m | 1 of 147 (0.7%) | 1 of 112 (0.9%) |
| next_depth (target) | 1 of 147 (0.7%) | 2 of 112 (1.8%) |

## 5. Checks (asserted in code)

- Scaler fitted on 952 readings, all dated on or before 2022-09-30: passed.
- Inverse transform reproduces depth_m, next_depth and the 5 weather features to 1e-9: passed.
- Raw columns of `c2_normalised.csv` are the C1 text unchanged (written from the string copy): passed by construction.

## 6. Files

- `preprocessing/c2_normalise.py`, `c2_scaler.json`, `c2_normalised.csv`, `c2_report.md`.
