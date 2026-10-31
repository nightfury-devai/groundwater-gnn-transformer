# C4 report: chronological 70/15/15 split

Samples are split by the date of their TARGET reading (`next_date`). No shuffling. Input: `preprocessing/c1_features.csv` (read only). The cutoffs below are used by C2 (scaler fit) and C3 (window split).

## 1. Cutoffs

- Train/val cutoff: **2022-09-30** (a train sample has target date on or before this).
- Val/test cutoff: **2023-12-31** (a val sample has target date after the train cutoff and on or before this; a test sample is after it).
- Both are quarter ends. They were chosen as the quarter ends whose cumulative share of the 840 GAP samples (by target date) is closest to 70% and 85%.
- Saved to `preprocessing/c4_split_dates.json`.

## 2. Samples per split

| variant | split | samples | share |
|---|---|---|---|
| GAP (gap-tolerant) | train | 581 | 69.2% |
| GAP (gap-tolerant) | val | 147 | 17.5% |
| GAP (gap-tolerant) | test | 112 | 13.3% |
| GAP (gap-tolerant) | total | 840 | 100% |
| STRICT (8 consecutive quarters + next-quarter target) | train | 80 | 51.0% |
| STRICT (8 consecutive quarters + next-quarter target) | val | 26 | 16.6% |
| STRICT (8 consecutive quarters + next-quarter target) | test | 51 | 32.5% |
| STRICT (8 consecutive quarters + next-quarter target) | total | 157 | 100% |

- STRICT uses the GAP cutoffs unchanged, so its shares differ from 70/15/15 (all STRICT windows are a subset of the GAP ones; the test period is identical in both).

## 3. Wells and districts per split

| variant | split | wells | districts | per district |
|---|---|---|---|---|
| GAP | train | 42 | 3 | Nagapattinam 229 samples / 14 wells, Thanjavur 266 samples / 21 wells, Thiruvarur 86 samples / 7 wells |
| GAP | val | 34 | 3 | Nagapattinam 59 samples / 13 wells, Thanjavur 56 samples / 14 wells, Thiruvarur 32 samples / 7 wells |
| GAP | test | 32 | 3 | Nagapattinam 51 samples / 13 wells, Thanjavur 42 samples / 12 wells, Thiruvarur 19 samples / 7 wells |
| STRICT | train | 16 | 2 | Nagapattinam 55 samples / 9 wells, Thanjavur 25 samples / 7 wells |
| STRICT | val | 13 | 3 | Nagapattinam 10 samples / 5 wells, Thanjavur 10 samples / 5 wells, Thiruvarur 6 samples / 3 wells |
| STRICT | test | 15 | 2 | Nagapattinam 30 samples / 8 wells, Thanjavur 21 samples / 7 wells |

## 4. Target-date range per split

| variant | split | first target date | last target date |
|---|---|---|---|
| GAP | train | 2015-05-14 | 2022-08-30 |
| GAP | val | 2022-11-30 | 2023-11-10 |
| GAP | test | 2024-01-10 | 2024-11-30 |
| STRICT | train | 2018-05-04 | 2020-01-09 |
| STRICT | val | 2023-08-30 | 2023-11-10 |
| STRICT | test | 2024-01-10 | 2024-11-30 |

## 5. Checks (asserted in code)

- GAP: train targets <= train cutoff; val targets in (train cutoff, val cutoff]; test targets > val cutoff; train < val < test in time: passed.
- STRICT: train targets <= train cutoff; val targets in (train cutoff, val cutoff]; test targets > val cutoff; train < val < test in time: passed.
- Every STRICT sample is also a GAP sample: passed.
- Cutoffs are quarter ends: passed.
- Nothing was shuffled; the order is well, then date (stable sort).

## 6. Files

- `preprocessing/c4_split.py`, `preprocessing/c4_split_dates.json`, `preprocessing/c4_report.md`.
