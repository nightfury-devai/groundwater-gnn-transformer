# D report: graph construction (distance matrix, adjacency, normalised adjacency)

Input: `graph/distance_matrix.npy` and `graph/well_order.json` (written by `graph/distance_matrix.py`). Arrays written: `adjacency.npy`, `a_hat.npy`, `graph_meta.json`. No model code was built.

## 1. Node order

- 53 nodes. Order rule: ascending well_id, the same order `preprocessing/c3_windows.py` iterates the wells (`groupby(sort=False)` on `c2_normalised.csv` sorted by well_id, date). `c3_windows_gap.npz` stores only a per-window `well_id` (42 wells appear there), no 53-node index, so the order was derived from the c3 code and the 42 npz wells were checked to appear in this same relative order (done in `distance_matrix.py`).
- Districts: Thanjavur 26, Nagapattinam 16, Thiruvarur 11.
- Haversine, R = 6371.0088 km. Off-diagonal distance: min 0.0000 km, median 49.137 km, mean 50.436 km, std 24.798 km, max 125.312 km.

## 2. Sigma used for the saved arrays

- sigma = **24.7975 km** (default: std of off-diagonal pairwise distances (population std)).
- Candidates (not chosen by the script): std of off-diagonal distances 24.7975 km, median 49.1370 km, 25 km, 10 km.

## 3. Sigma candidates

Neighbours = other wells j with A_ij above the threshold (off-diagonal, per well). A_ij > 0.5 means d < 0.8326 sigma; A_ij > 0.1 means d < 1.5174 sigma.

| candidate | sigma (km) | neighbours A>0.5 (mean / median) | neighbours A>0.1 (mean / median) | wells with 0 neighbours at A>0.1 | cross-district share of A weight | A_hat eigenvalues min / max |
|---|---|---|---|---|---|---|
| std of off-diagonal distances | 24.798 | 6.49 / 6.0 | 17.66 / 18.0 | 0 | 35.7% | -0.0000 / 1.0000 |
| median off-diagonal distance | 49.137 | 20.30 / 21.0 | 42.45 / 43.0 | 0 | 50.5% | -0.0000 / 1.0000 |
| 25 km | 25.000 | 6.53 / 6.0 | 17.85 / 18.0 | 0 | 35.9% | -0.0000 / 1.0000 |
| 10 km | 10.000 | 1.09 / 1.0 | 3.77 / 4.0 | 2 | 22.1% | -0.0000 / 1.0000 |

### 3a. Five most isolated wells per candidate (smallest row sum of A, diagonal 1 included)

**std of off-diagonal distances (sigma 24.798 km):**

| rank | well | district | row sum of A |
|---|---|---|---|
| 1 | Peravurani | Thanjavur | 2.9325 |
| 2 | Vedaranyam | Nagapattinam | 3.0661 |
| 3 | Vedaranyam1 | Nagapattinam | 3.0661 |
| 4 | Puttur | Nagapattinam | 4.3910 |
| 5 | Thagattur | Nagapattinam | 4.3939 |

**median off-diagonal distance (sigma 49.137 km):**

| rank | well | district | row sum of A |
|---|---|---|---|
| 1 | Vedaranyam | Nagapattinam | 10.1368 |
| 2 | Vedaranyam1 | Nagapattinam | 10.1368 |
| 3 | Peravurani | Thanjavur | 11.7644 |
| 4 | Puttur | Nagapattinam | 13.9225 |
| 5 | Sirgazhi1 | Nagapattinam | 15.4738 |

**25 km (sigma 25.000 km):**

| rank | well | district | row sum of A |
|---|---|---|---|
| 1 | Peravurani | Thanjavur | 2.9747 |
| 2 | Vedaranyam | Nagapattinam | 3.0947 |
| 3 | Vedaranyam1 | Nagapattinam | 3.0947 |
| 4 | Puttur | Nagapattinam | 4.4530 |
| 5 | Thagattur | Nagapattinam | 4.4574 |

**10 km (sigma 10.000 km):**

| rank | well | district | row sum of A |
|---|---|---|---|
| 1 | Peravurani | Thanjavur | 1.0514 |
| 2 | Tiruvonam | Thanjavur | 1.2338 |
| 3 | Pattukottai1 | Thanjavur | 1.2419 |
| 4 | Madukkur | Thanjavur | 1.3499 |
| 5 | Thagattur | Nagapattinam | 1.3537 |

## 4. Checks on the saved arrays (sigma used)

- PASS: distance matrix symmetric
- PASS: distance matrix diagonal = 0
- PASS: A symmetric
- PASS: A diagonal = 1
- PASS: all A values in (0, 1] (min 8.120e-12, max 1.0)
- PASS: D has no zeros (row sums of A: min 2.9325, max 13.1615)
- PASS: A_hat symmetric (max |A_hat - A_hat^T| = 2.78e-17)
- PASS: eigenvalues of A_hat within [-1, 1] (min -0.000000, max 1.000000)
- PASS: A_hat equals D^(-1/2) A D^(-1/2) elementwise

## 5. Hand-calculated haversine spot checks

Independent pure-Python haversine (atan2 form, `math` module) against the matrix entry.

- PASS: closest non-zero pair: matrix = hand haversine (Paramasivapuram II - Paramasivapuram III: matrix 0.154887 km, hand 0.154887 km)
- PASS: closest non-zero pair: expected wells and distance (expected ['Paramasivapuram II', 'Paramasivapuram III'] about 0.15 km; found Paramasivapuram II / Paramasivapuram III at 0.155 km)
- PASS: farthest pair: matrix = hand haversine (Peravurani - Puttur: matrix 125.311501 km, hand 125.311501 km)
- PASS: farthest pair: expected wells and distance (expected ['Peravurani', 'Puttur'] about 125.3 km; found Peravurani / Puttur at 125.312 km)
- PASS: closest cross-district pair: matrix = hand haversine (Paramasivapuram III - Peralam: matrix 3.504386 km, hand 3.504386 km)

| pair | well 1 (district) | well 2 (district) | matrix d (km) | hand d (km) | A_ij (sigma used) |
|---|---|---|---|---|---|
| closest non-zero pair | Paramasivapuram II (Nagapattinam) | Paramasivapuram III (Nagapattinam) | 0.1549 | 0.1549 | 0.999961 |
| farthest pair | Peravurani (Thanjavur) | Puttur (Nagapattinam) | 125.3115 | 125.3115 | 0.000000 |
| closest cross-district pair | Paramasivapuram III (Nagapattinam) | Peralam (Thiruvarur) | 3.5044 | 3.5044 | 0.980227 |

## 6. Pairs at distance exactly 0 (kept as separate nodes, not merged, dropped or jittered)

| well 1 | well 2 | district(s) | lat | lon | d (km) | A_ij |
|---|---|---|---|---|---|---|
| Icchankottai | Icchankottai1 | Thanjavur | 10.650000 | 79.177780 | 0 | 1.0 |
| Vedaranyam | Vedaranyam1 | Nagapattinam | 10.350000 | 79.862500 | 0 | 1.0 |

- 2 pair(s). Expected Icchankottai / Icchankottai1 and Vedaranyam / Vedaranyam1: matches.

## 7. Cross-district share of off-diagonal A weight (sigma used)

- Total off-diagonal A weight 430.534; between wells of different districts 153.795; **share 35.72%**.

| off-diagonal A weight, row district to column district | Thanjavur | Nagapattinam | Thiruvarur |
|---|---|---|---|
| Thanjavur | 180.699 | 17.554 | 36.838 |
| Nagapattinam | 17.554 | 55.589 | 22.505 |
| Thiruvarur | 36.838 | 22.505 | 40.451 |

## 8. Files

- `graph/distance_matrix.py`, `graph/graph_builder.py`, `graph/distance_matrix.npy`, `graph/well_order.json`, `graph/adjacency.npy`, `graph/a_hat.npy`, `graph/graph_meta.json`, `graph/d_report.md`.

Overall: ALL CHECKS PASSED
