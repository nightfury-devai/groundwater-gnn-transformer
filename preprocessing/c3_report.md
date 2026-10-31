# C3 report: sliding windows (L = 8)

Input `preprocessing/c2_normalised.csv` (read only). A window = 8 readings of one well ending at reading k; target = depth of reading k+1 of the same well. Split by target date with the C4 cutoffs (train <= 2022-09-30, val <= 2023-12-31, test after). X = the 6 normalised features; G = gap in quarters before each position; shapes below.

## 1. Counts against the earlier hand counts

| variant | windows | wells | earlier hand count | result |
|---|---|---|---|---|
| GAP | 840 | 42 | about 840 windows / 42 wells | matches |
| STRICT (8 consecutive quarters + target in the next quarter) | 157 | 23 | about 157 windows / 23 wells | matches |
| for reference: 8 consecutive quarters, target gap unconstrained (not saved) | 180 | 25 | - | - |

- GAP reproduces the earlier count exactly.
- STRICT reproduces the earlier count exactly once the target must be in the very next quarter (decision 2026-10-18). The earlier hand count had that rule; the first version of this script did not and gave 180 windows from 25 wells, so the 23 windows whose target is 2 or more quarters after the last input were removed.

Array shapes: GAP X (840, 8, 6), G (840, 8); STRICT X (157, 8, 6), G (157, 8). Saved: `c3_windows_gap.npz`, `c3_windows_strict.npz` (arrays: X, G, target_gap, y, y_metres, well_id, district, last_input_date, target_date, split; no pickle).
- `G[:, 0]` is NaN when the window starts at the first reading of its well (nothing before it): GAP 42 windows, STRICT 0 windows. `G[:, 1:]` is always defined. Left as NaN, not filled.

## 2. Windows and wells per split

| variant | split | windows | wells | first target | last target |
|---|---|---|---|---|---|
| GAP | train | 581 | 42 | 2015-05-14 | 2022-08-30 |
| GAP | val | 147 | 34 | 2022-11-30 | 2023-11-10 |
| GAP | test | 112 | 32 | 2024-01-10 | 2024-11-30 |
| STRICT | train | 80 | 16 | 2018-05-04 | 2020-01-09 |
| STRICT | val | 26 | 13 | 2023-08-30 | 2023-11-10 |
| STRICT | test | 51 | 15 | 2024-01-10 | 2024-11-30 |

## 3. Windows per district

| variant | split | Thanjavur | Nagapattinam | Thiruvarur |
|---|---|---|---|---|
| GAP | train | 266 / 21 wells | 229 / 14 wells | 86 / 7 wells |
| GAP | val | 56 / 14 wells | 59 / 13 wells | 32 / 7 wells |
| GAP | test | 42 / 12 wells | 51 / 13 wells | 19 / 7 wells |
| GAP | all | 364 / 21 wells | 339 / 14 wells | 137 / 7 wells |
| STRICT | train | 25 / 7 wells | 55 / 9 wells | 0 / 0 wells |
| STRICT | val | 10 / 5 wells | 10 / 5 wells | 6 / 3 wells |
| STRICT | test | 21 / 7 wells | 30 / 8 wells | 0 / 0 wells |
| STRICT | all | 56 / 9 wells | 95 / 11 wells | 6 / 3 wells |

- A STRICT split with no window in a district (see the table) means that district contributes nothing to training in that variant.

## 4. Gap from the last input to the target (quarters)

| target gap | GAP windows | STRICT windows |
|---|---|---|
| 1 | 652 (77.6%) | 157 (100.0%) |
| 2 | 115 (13.7%) | 0 (0.0%) |
| 3 | 53 (6.3%) | 0 (0.0%) |
| 4 | 9 (1.1%) | 0 (0.0%) |
| 5 | 3 (0.4%) | 0 (0.0%) |
| 6 | 2 (0.2%) | 0 (0.0%) |
| 7 | 1 (0.1%) | 0 (0.0%) |
| 8 | 2 (0.2%) | 0 (0.0%) |
| 11 | 2 (0.2%) | 0 (0.0%) |
| 21 | 1 (0.1%) | 0 (0.0%) |

## 5. Span of an input window (first to last input reading, days)

| variant | median | min | max | mean |
|---|---|---|---|---|
| GAP | 912 | 590 | 2938 | 980 |
| STRICT | 639 | 590 | 694 | 641 |

A strict 8-quarter window spans about 7 quarters (about 640 days); GAP windows can span many years where readings are sparse.

## 6. Persistence baseline (prediction = depth_m at the last input reading), MAE in metres

Descriptive baseline, not a model. Error = |depth_m(last input) - next depth|, on the raw metre values.

| variant | split | windows | MAE (m) | RMSE (m) |
|---|---|---|---|---|
| GAP | train | 581 | 1.887 | 2.647 |
| GAP | val | 147 | 0.888 | 1.338 |
| GAP | test | 112 | 1.347 | 1.826 |
| STRICT | train | 80 | 2.020 | 2.757 |
| STRICT | val | 26 | 1.056 | 1.515 |
| STRICT | test | 51 | 1.346 | 1.777 |

## 7. Negative Allankottai depths (kept as they are)

- Windows whose target is negative: GAP 3, STRICT 0. Windows with a negative depth among the 8 inputs: GAP 10, STRICT 0.

## 8. Checks (asserted in code)

- GAP: target_date > last_input_date for all 840 windows: passed.
- GAP: train targets <= 2022-09-30, val/test targets after it, val <= 2023-12-31 < test: passed.
- GAP: wells are contiguous blocks, windows in strictly increasing time inside each well (no mixing, no shuffling): passed.
- GAP: all 840 X blocks and targets re-read independently from their own well rows: equal: passed.
- STRICT: target_date > last_input_date for all 157 windows: passed.
- STRICT: train targets <= 2022-09-30, val/test targets after it, val <= 2023-12-31 < test: passed.
- STRICT: wells are contiguous blocks, windows in strictly increasing time inside each well (no mixing, no shuffling): passed.
- STRICT: all 157 X blocks and targets re-read independently from their own well rows: equal: passed.
- STRICT windows are a subset of GAP windows: passed.
- scaler (C2) statistics recomputed from readings dated <= 2022-09-30 (952 readings): equal to c2_scaler.json: passed.
- Re-running gives identical files: the npz files are written with a fixed zip timestamp and the CSV/JSON/MD outputs contain no timestamps. File fingerprints (first 16 hex of SHA-256) for comparison between runs: gap `c4271fbafbc2ac06`, strict `0cf5db06c596231a`.

## 9. Files

- `preprocessing/c3_windows.py`, `c3_windows_gap.npz`, `c3_windows_strict.npz`, `c3_report.md`.
