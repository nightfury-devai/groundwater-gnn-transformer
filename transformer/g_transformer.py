"""
Phase G1: Transformer temporal encoder. Defined here, nothing is trained.

    H0 = X W_e + b_e + PE                      X (B, L, 6) -> H0 (B, L, 64)     (same as Phase E)
    Hn = Encoder(H0)                           2 pre-norm layers + final LayerNorm -> (B, L, 64)
    h_target = Hn[:, -1, :]                    the latest reading, (B, 64)  (goes to the late fusion in Phase H)

Design (final, decided 2026-10-29):
  - nn.TransformerEncoder of 2 x nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.2,
    activation="gelu", batch_first=True, norm_first=True) and a final nn.LayerNorm(64) (the `norm` argument).
  - enable_nested_tensor=False: the nested-tensor fast path does not support norm_first=True, and there is no padding to skip.
  - NO attention mask: all L positions are readings dated on or before the forecast time, so every position may look at
    every other one (no future information exists in the window).
  - The positional encoding is a fixed, non-trainable buffer. For L = 8 it is loaded from encoding/pe_table.npy (Phase E) and
    asserted equal to the formula; for any other L it is built with the same formula (`positional_encoding`).
  - L is a constructor argument (default 8) so Phase L can use 4 or 12.
The PE counts readings (positions), not time: the gap array G is not used here.
"""

import contextlib
import os

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PE_PATH = f"{ROOT}/encoding/pe_table.npy"


def positional_encoding(length, d):
    """Sinusoidal table (length, d): PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(same angle). float64."""
    if d % 2:
        raise ValueError("d_model must be even")
    pos = np.arange(length)[:, None]
    i = np.arange(d // 2)[None, :]
    angle = pos / np.power(10000.0, 2 * i / d)
    pe = np.zeros((length, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


def expected_parameter_counts(in_dim=6, d_model=64, ff=128, n_layers=2):
    """Parameter counts from the formulas (no module needed). In-projection and out-projection of the attention,
    two feed-forward matrices, two LayerNorms per layer, one final LayerNorm, the embedding."""
    attn = d_model * 3 * d_model + 3 * d_model + d_model * d_model + d_model          # 16,640 for d=64
    ffn = d_model * ff + ff + ff * d_model + d_model                                   # 16,576 for ff=128
    norms = 2 * (2 * d_model)                                                          # 256
    layer = attn + ffn + norms                                                         # 33,472
    embed = in_dim * d_model + d_model                                                 # 448
    encoder = n_layers * layer + 2 * d_model                                           # + final LayerNorm (128)
    return {"embedding": embed, "attention_per_layer": attn, "feed_forward_per_layer": ffn, "norms_per_layer": norms,
            "layer": layer, "layers_total": n_layers * layer, "final_norm": 2 * d_model, "encoder": encoder,
            "total": embed + encoder}


@contextlib.contextmanager
def no_fastpath():
    """Switch PyTorch's fused inference 'fast path' for attention / encoder layers off, then restore the old setting.

    In eval mode under torch.no_grad() nn.TransformerEncoderLayer silently uses a fused kernel. On this machine's CUDA
    build that kernel differs from the ordinary (training) arithmetic by about 3e-4 (measured in transformer/g_report.md),
    so validation and test outputs would come from slightly different math than training. The module therefore always
    uses the ordinary path: eval and train results are computed the same way and CPU and GPU agree to about 1e-6."""
    prev = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)


class TemporalEncoder(nn.Module):
    def __init__(self, L=8, in_dim=6, d_model=64, nhead=4, dim_feedforward=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.L, self.in_dim, self.d_model = L, in_dim, d_model
        self.embed = nn.Linear(in_dim, d_model)                                       # X W_e + b_e
        pe = positional_encoding(L, d_model)
        if L == 8:
            # Phase E table is the reference: load it and make sure it is the same table the formula gives
            if not os.path.exists(PE_PATH):
                raise FileNotFoundError(f"{PE_PATH} not found; run run_phase_e.bat first")
            table = np.load(PE_PATH)
            if table.shape != pe.shape or np.abs(table - pe).max() > 1e-12:
                raise ValueError("encoding/pe_table.npy does not match the PE formula")
            pe = table
        self.register_buffer("pe", torch.tensor(pe, dtype=torch.float32))              # not a parameter, never trained
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)

    def forward(self, x, return_attention=False):
        """x (B, L, in_dim). Returns (Hn, h_target) or (Hn, h_target, attn) with attn = list over layers of
        (B, heads, L, L) weights (per head, not averaged). Attention weights are only meaningful in eval mode
        (in train mode PyTorch returns them with attention dropout applied)."""
        if x.dim() != 3 or x.shape[1] != self.L or x.shape[2] != self.in_dim:
            raise ValueError(f"expected x of shape (B, {self.L}, {self.in_dim}), got {tuple(x.shape)}")
        h = self.embed(x) + self.pe                                                    # H0, PE broadcast over the batch
        attn = []
        # The layers are applied one by one (identical to nn.TransformerEncoder.forward, which only loops over
        # the layers and applies the final norm) so each layer's attention weights can be read out on the way.
        with no_fastpath():                                                            # same arithmetic in train and eval
            for lay in self.encoder.layers:
                if return_attention:
                    xn = lay.norm1(h)                                                  # pre-norm: attention sees norm1(h)
                    _, w = lay.self_attn(xn, xn, xn, need_weights=True, average_attn_weights=False)
                    attn.append(w)
                h = lay(h)
            hn = self.encoder.norm(h)
        h_target = hn[:, -1, :]                                                        # latest reading
        return (hn, h_target, attn) if return_attention else (hn, h_target)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())
