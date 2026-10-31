# F report: GCN spatial encoder (F1 node features, F2 encoder, F3 checks)

Nothing is trained. Node features use only readings dated on or before each window's last input date.

## F1. Node features (last observation carried forward)

- Windows 840, nodes 53 (from `graph/well_order.json`), readings used from `c2_normalised.csv`: 1211. Feature order read from `preprocessing/c3_windows.py`: rainfall_180d_avg_mm, tmax_90d_avg_c, tmin_90d_avg_c, humidity_90d_avg_pct, et_90d_sum_mm, depth_m.
- Node vector = 6 features + staleness + has_reading = 8 values. Staleness = days since the carried reading / 91.25, then z-scored with train-only mean 4.6595 and std 8.3069 (over 30421 observed node values in 581 train windows), saved in `gnn/f_staleness_scaler.json`.
- Output: `gnn/f_node_features.npz` with NF, node_idx, well_id, split, last_input_date.

- PASS: NF shape is (windows, nodes, 8) ((840, 53, 8))
- PASS: NF has no NaN (0 NaN)
- PASS: every reading used has date <= last_input_date (max (date used - t) = 0 days)
- PASS: independent brute-force search finds the same reading for all window-node pairs (0 differences in 44520 pairs)
- PASS: unobserved nodes are all-zero (features, staleness, has_reading) (373 unobserved window-node pairs)
- PASS: NF[w, node_idx[w], :6] equals X[w, last position, :] (max abs difference 1.72e-07 (float32 storage))
- PASS: target well is observed with staleness 0 in every window
- PASS: node_idx maps every window's well_id to its position in well_order.json

Observed nodes per window (has_reading = 1) and staleness of the carried readings, by split:

| split | windows | observed nodes per window (min / median / max) | staleness in quarters (min / median / max, observed nodes) |
|---|---|---|---|
| train | 581 | 52 / 52 / 53 | 0.00 / 0.00 / 37.65 |
| val | 147 | 52 / 53 / 53 | 0.00 / 0.00 / 42.66 |
| test | 112 | 53 / 53 / 53 | 0.00 / 0.00 / 46.67 |

Wells with no window of their own: 11 of 53. Number of windows in which each is still observed as a neighbour node:

| well | district | readings | windows in which observed | share of windows |
|---|---|---|---|---|
| Aduthurai | Thanjavur | 5 | 840 | 100.0% |
| Ayyampet | Thanjavur | 1 | 840 | 100.0% |
| Kodavasal | Thiruvarur | 3 | 840 | 100.0% |
| Madukkur | Thanjavur | 6 | 840 | 100.0% |
| Nagapattinam1 | Nagapattinam | 1 | 840 | 100.0% |
| Tirutturaipoondi | Thiruvarur | 1 | 840 | 100.0% |
| Tiruvarur | Thiruvarur | 6 | 840 | 100.0% |
| Tiruvonam | Thanjavur | 3 | 840 | 100.0% |
| Vadavur | Thanjavur | 1 | 840 | 100.0% |
| Valangiman | Thiruvarur | 4 | 840 | 100.0% |
| Vedaranyam1 | Nagapattinam | 4 | 467 | 55.6% |

- PASS: each well without a window is read from the table (count taken from the files) (11 wells)

## F2. GCN encoder

- `GCNEncoder`: in_dim 8, hidden 64, out 64, 2 layers, dropout 0.2, H1 = ReLU(A_hat (H0 W1) + b1), H2 = ReLU(A_hat (H1 W2) + b2); A_hat loaded from `graph/a_hat.npy` (not rebuilt), no extra identity; bias added after the A_hat product; Glorot initialisation, seed 42.

## F3. Encoder checks

- PASS: Z shape is (4, 53, 64) for 4 train windows ((4, 53, 64))
- PASS: z_target shape is (4, 64) ((4, 64))
- PASS: no NaN in Z
- PASS: z_target equals Z[b, node_idx[b], :]
- PASS: parameter count = 8*64 + 64 + 64*64 + 64 (4736 (expected 4736))
- PASS: module output equals the plain numpy two-layer calculation (eval mode) (max abs difference 2.49e-07)
- PASS: A_hat in the module equals graph/a_hat.npy exactly (float32 copy) (max difference to the float64 file 7.41e-09 (float32 rounding))
- PASS: A_hat symmetric (max asymmetry 2.78e-17)
- PASS: A_hat row sums within [0.80, 1.10] (min 0.8054, max 1.0948)

- PASS: adding 1.0 to feature 0 of the nearest neighbour changes z_target more than adding it to the farthest well (Keelvalur window 39; nearest Nagapattinam1 (A_hat 0.0984): change 0.0420; farthest Puduchataram (A_hat 8.17e-07): change 1.05e-04)

Change in z_target (L2 norm) when +1.0 is added to one input feature of one node, same window, eval mode (information; the check above uses feature 0):

| node | feature 0 | feature 1 | feature 2 | feature 3 | feature 4 | feature 5 | feature 6 | feature 7 |
|---|---|---|---|---|---|---|---|---|
| nearest: Nagapattinam1 | 0.0420 | 0.0408 | 0.0471 | 0.0356 | 0.0438 | 0.0571 | 0.0324 | 0.0527 |
| farthest: Puduchataram | 1.05e-04 | 1.18e-04 | 1.39e-04 | 1.27e-04 | 9.94e-05 | 1.49e-04 | 8.23e-05 | 1.15e-04 |

Effective reach of 2 layers, A2 = A_hat @ A_hat (information for the paper, not an assert). Self weight A2[i,i] over the 53 wells:

| statistic | value |
|---|---|
| min | 0.0449 (Valangiman) |
| median | 0.0606 |
| max | 0.2365 (Vedaranyam1) |
| wells with > 50% of their A2 row weight outside themselves | 53 of 53 |

Five smallest self weights: Valangiman 0.0449, Kodavasal 0.0454, Needamangalam 0.0458, Papanasam 0.0459, Mannargudi 0.0468.
Five largest self weights: Vedaranyam1 0.2365, Vedaranyam 0.2365, Peravurani 0.1629, Thagattur 0.1392, Thillaivilagam 0.1359.

- PASS: A2 self weights computed for every well (information) (53 of 53 wells have > 50% outside weight)
- PASS: train mode: two forward passes differ (dropout active) (max difference 0.292)
- PASS: eval mode: two forward passes are identical (max difference 0.0)
- PASS: CPU and CUDA outputs agree within 1e-4 (device NVIDIA GeForce RTX 3050 6GB Laptop GPU, max abs difference 1.79e-07)

## Files

- `gnn/f_node_features.py`, `gnn/f_node_features.npz`, `gnn/f_staleness_scaler.json`, `gnn/f_gcn.py`, `gnn/run_phase_f.py`, `gnn/f_report.md`, `gnn/f_results.json`, `run_phase_f.bat`.

ALL CHECKS PASSED
