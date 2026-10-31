# E report: temporal encoding (E1 sequences, E2 positional encoding, E3 embedding + PE)

Nothing is trained. Gap information (quarters between readings, array G) is stored in the windows but is NOT used by this encoding yet.

## E1. Per-well sequences X_i of shape [L x F]

Keys and shapes of `preprocessing/c3_windows_gap.npz`:

| key | shape | dtype |
|---|---|---|
| G | (840, 8) | float64 |
| X | (840, 8, 6) | float64 |
| district | (840,) | <U12 |
| last_input_date | (840,) | <U10 |
| split | (840,) | <U5 |
| target_date | (840,) | <U10 |
| target_gap | (840,) | int64 |
| well_id | (840,) | <U19 |
| y | (840,) | float64 |
| y_metres | (840,) | float64 |

- PASS: X shape is (840, 8, 6) ((840, 8, 6))
- PASS: target y has 840 values and y_metres matches
- PASS: split counts 581 train / 147 val / 112 test ({'train': 581, 'val': 147, 'test': 112})
- PASS: 42 wells (42)
- PASS: no NaN in X (inputs) (0 NaN)
- PASS: G[:,0] is NaN in 42 windows (42 NaN)
- PASS: G[:,1:] has no NaN (all NaN in G are in column 0) (total 42, column 0 42, columns 1-7 0)
- PASS: X is finite (min -2.474, max 4.159)
- PASS: split labels agree with c4_split_dates.json cutoffs (train <= 2022-09-30, val <= 2023-12-31, test after)

## E2. Sinusoidal positional encoding (numpy only)

- PASS: PE table shape is (8, 64), saved to encoding/pe_table.npy ((8, 64))
- PASS: PE[0,0] = 0 and PE[0,1] = 1 (0.0, 1.0)
- PASS: all PE values in [-1, 1] (min -0.9900, max 1.0000)
- PASS: all 8 rows are different
- PASS: 3 entries match independent math.sin / math.cos (PE[1,0] table 0.8414709848 vs hand 0.8414709848; PE[3,11] table 0.7574406581 vs hand 0.7574406581; PE[7,62] table 0.0009334649 vs hand 0.0009334649)

## E3. Linear embedding to d_h = 64, then add PE: H0 = (X W_e + b_e) + PE

- torch 2.6.0+cu124, CUDA available: True
- PASS: output shape is (4, 8, 64) ((4, 8, 64))
- PASS: parameter count 6*64 + 64 = 448 (448)
- PASS: output equals (X @ W^T + b) + PE by hand within 1e-5 (max abs diff 2.44e-07)

## Gap information

- The gap array G (quarters between consecutive readings, NaN in column 0 for 42 windows) is stored in the windows but is not used by this encoding yet.

## Files

- `encoding/e_temporal.py`, `encoding/pe_table.npy`, `encoding/e_report.md`, `encoding/e_results.json`, `encoding/run_phase_e.py`.

ALL CHECKS PASSED
