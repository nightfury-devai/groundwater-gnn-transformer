"""
Phase F2: two-layer GCN spatial encoder (Kipf and Welling propagation). Defined here, nothing is trained.

    H1 = ReLU( A_hat (H0 W1) + b1 ),   dropout 0.2 (active in train mode only)
    H2 = ReLU( A_hat (H1 W2) + b2 )

A_hat comes from graph/a_hat.npy (Phase D) and is NOT rebuilt or modified. It already carries a self
weight on its diagonal (the Gaussian kernel gives A_ii = 1 before normalisation), so NO extra identity is
added here (Kipf and Welling's "A + I" is already inside A_hat). W1, W2 are bias-free matrices; the biases b1, b2
are separate parameters added AFTER the A_hat multiplication.

Shapes: H0 (B, 53, 8) -> Z (B, 53, 64). `target_embedding(Z, node_idx)` returns Z[b, node_idx[b], :], shape (B, 64);
this row is what goes to the late fusion in Phase H.

Parameters: in*hidden + hidden + hidden*out + out = 8*64 + 64 + 64*64 + 64 = 4736.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A_HAT_PATH = f"{ROOT}/graph/a_hat.npy"
WELL_ORDER_PATH = f"{ROOT}/graph/well_order.json"


def load_a_hat(path=A_HAT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"graph/a_hat.npy not found at {path}; run run_phase_d.bat first")
    a = np.load(path)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"A_hat must be square, got {a.shape}")
    return a


def load_well_order(path=WELL_ORDER_PATH):
    with open(path) as f:
        return [w["well_id"] for w in json.load(f)["wells"]]


class GCNEncoder(nn.Module):
    def __init__(self, a_hat=None, in_dim=8, hidden=64, out_dim=64, dropout=0.2):
        super().__init__()
        a = load_a_hat() if a_hat is None else np.asarray(a_hat)
        # float32 buffer: moves with .to(device), is saved in state_dict, never trained
        self.register_buffer("a_hat", torch.tensor(a, dtype=torch.float32))
        self.n_nodes = self.a_hat.shape[0]
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        # bias-free weight matrices and separate biases (bias added after the A_hat product)
        self.W1 = nn.Parameter(torch.empty(in_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.empty(hidden, out_dim))
        self.b2 = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.W1)        # Glorot initialisation as in Kipf and Welling
        nn.init.xavier_uniform_(self.W2)
        self.dropout = nn.Dropout(dropout)      # only acts in train mode

    def forward(self, h0):
        if h0.dim() != 3 or h0.shape[1] != self.n_nodes or h0.shape[2] != self.in_dim:
            raise ValueError(f"expected H0 of shape (B, {self.n_nodes}, {self.in_dim}), got {tuple(h0.shape)}")
        # (53,53) @ (B,53,64): matmul broadcasts A_hat over the batch
        h1 = torch.relu(torch.matmul(self.a_hat, h0 @ self.W1) + self.b1)
        h1 = self.dropout(h1)
        return torch.relu(torch.matmul(self.a_hat, h1 @ self.W2) + self.b2)

    @staticmethod
    def target_embedding(z, node_idx):
        """Row of the target well for every window: Z[b, node_idx[b], :] -> (B, 64)."""
        node_idx = torch.as_tensor(node_idx, dtype=torch.long, device=z.device)
        return z[torch.arange(z.shape[0], device=z.device), node_idx, :]

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())
