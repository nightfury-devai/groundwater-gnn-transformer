# G report: Transformer temporal encoder (G1 module, G2 checks)

Nothing is trained. Design (decided 2026-10-29): embedding Linear(6, 64) + fixed sinusoidal PE; 2 x TransformerEncoderLayer(d_model 64, 4 heads, feed-forward 128, dropout 0.2, GELU, batch_first, pre-norm) + final LayerNorm(64); `enable_nested_tensor=False` (pre-norm layers do not support it); no attention mask (every position is a reading on or before the forecast time); output h_target = Hn[:, -1, :], the latest reading.

## G1. Module and data

- Windows: 840 (train 581, val 147, test 112); each window is 8 readings x 6 standardised features. X has 0 NaN.

- Check batch (4 train windows, eval mode unless stated): [39, 0, 1, 2]; first = Keelvalur window 39 (a train window).

## G2. Checks

- PASS: the fused-kernel switch is restored after a forward pass
- PASS: input (4, 8, 6) -> H0 (4, 8, 64) -> Hn (4, 8, 64), h_target (4, 64)
- PASS: no NaN in H0, Hn, h_target
- PASS: h_target equals Hn[:, -1, :]
- PASS: module forward equals nn.TransformerEncoder's own forward (max abs difference 0.00e+00)
- PASS: parameter counts from the formula equal the module's counts (embedding 448, per layer 33472 = 16640 + 16576 + 256, two layers 66944, final LayerNorm 128, encoder 67072, total 67520)
- PASS: positional encoding is a buffer, not a parameter

| part | expected (formula) | module | status |
|---|---|---|---|
| embedding | 448 | 448 | PASS |
| attention_per_layer | 16640 | 16640 | PASS |
| feed_forward_per_layer | 16576 | 16576 | PASS |
| norms_per_layer | 256 | 256 | PASS |
| layer | 33472 | 33472 | PASS |
| layers_total | 66944 | 66944 | PASS |
| final_norm | 128 | 128 | PASS |
| encoder | 67072 | 67072 | PASS |
| total | 67520 | 67520 | PASS |

- PASS: module output equals the plain numpy encoder (embedding + PE, 2 pre-norm layers, final LayerNorm), eval mode (max abs difference 6.19e-07)
- PASS: attention: every row sums to 1 and all weights are >= 0 (all layers, all heads) (layers 2, shape per layer (4, 4, 8, 8), max |row sum - 1| = 1.79e-07, min weight 3.65e-02)
- PASS: module attention weights equal the numpy softmax(Q K^T / sqrt(d_k)) weights (max abs difference 5.87e-08)

Layer 1 attention of the last position (position 8 = latest reading) over the 8 positions, Keelvalur window 39, eval mode (information only: the weights are untrained, so they carry no meaning yet):

| head | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | pos 6 | pos 7 | pos 8 |
|---|---|---|---|---|---|---|---|---|
| head 1 | 0.091 | 0.064 | 0.142 | 0.203 | 0.075 | 0.119 | 0.118 | 0.187 |
| head 2 | 0.077 | 0.101 | 0.145 | 0.151 | 0.117 | 0.119 | 0.146 | 0.144 |
| head 3 | 0.073 | 0.096 | 0.106 | 0.102 | 0.180 | 0.149 | 0.170 | 0.122 |
| head 4 | 0.112 | 0.082 | 0.148 | 0.155 | 0.065 | 0.095 | 0.161 | 0.182 |

- PASS: without PE, permuting the first 7 readings leaves h_target unchanged (max diff < 1e-5) (max difference 4.77e-07; permutation [0, 1, 5, 2, 4, 3, 6, 7])
- PASS: with PE, the same permutation changes h_target (max diff > 1e-3) (max difference 0.123)
- PASS: train mode: two forward passes differ (dropout active) (max difference 1.290)
- PASS: eval mode: two forward passes are identical (max difference 0.0)
- PASS: loss = h_target.sum(): every parameter has a gradient that is not None and finite (28 parameter tensors; None: 0, non-finite: 0)
- PASS: loss = h_target . r (fixed random vector): every parameter tensor has a non-zero gradient (tensors with all-zero gradient: none)
- INFO: with loss = h_target.sum() the largest gradient inside the 2 encoder layers is 1.7e-07 (about 0): the sum of a LayerNorm output with weight 1 and bias 0 is 0 for any input, so that loss says nothing about the layers; the random-vector loss above is the informative gradient test.
- PASS: CPU and CUDA outputs agree within 1e-4 (device NVIDIA GeForce RTX 3050 6GB Laptop GPU, max abs difference 7.15e-07)
- PASS: CUDA output equals the float64 numpy encoder within 1e-5 (max abs difference 5.38e-07)
- INFO: PyTorch's fused eval-mode CUDA kernel (not used by the module) differs from the CPU result by 2.75e-04; the module switches it off so eval and train use the same arithmetic.
- PASS: PE(8, 64) from the formula equals encoding/pe_table.npy, and the module buffer is that table (max abs difference 0.0e+00)
- PASS: window length L = 4: Hn (4, 4, 64) and h_target (4, 64)
- PASS: window length L = 12: Hn (4, 12, 64) and h_target (4, 64)
- PASS: PE for L = 4 equals the first 4 rows of the L = 8 table (positions do not depend on L)
- PASS: Phase F windows are the same windows (well_id and last_input_date match)
- PASS: h_target size equals the GCN z_target size (ready for the late fusion in Phase H) (h_target (4, 64), z_target (4, 64))

- INFO: forward pass of a batch of 128 windows on NVIDIA GeForce RTX 3050 6GB Laptop GPU: 2.16 ms (mean of 20), peak GPU memory 11.2 MiB (varies slightly between runs)

## Files

- `transformer/g_transformer.py`, `transformer/run_phase_g.py`, `transformer/g_report.md`, `transformer/g_results.json`, `run_phase_g.bat`.

ALL CHECKS PASSED
