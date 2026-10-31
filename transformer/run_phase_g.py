"""
Phase G runner: builds the Transformer temporal encoder (G1) and runs the checks (G2). Nothing is trained.
Writes transformer/g_report.md and transformer/g_results.json, prints PASS/FAIL lines and ends with
ALL CHECKS PASSED or the list of failed checks.

    .\\run_phase_g.bat            (= .venv\\Scripts\\python.exe -B -I transformer\\run_phase_g.py)

numpy and torch only (scipy is blocked on this PC). All random seeds = 42.
The timing / memory numbers (check 11) are information only and change a little from run to run.
"""

import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)                       # -I removes the script folder from sys.path
sys.path.insert(0, os.path.join(ROOT, "gnn"))
import g_transformer as gt                     # noqa: E402

SEED = 42
KEELVALUR = "Keelvalur"
N_BATCH = 4
L, CHECKS, RES = [], [], {}


def say(s=""):
    L.append(s)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def check(name, ok, detail=""):
    st = "SKIPPED" if ok is None else ("PASS" if ok else "FAIL")
    CHECKS.append((name, st, detail))
    say(f"- {st}: {name}" + (f" ({detail})" if detail else ""))


def info(text):
    say(f"- INFO: {text}")


# ----------------------------------------------------------------- plain numpy re-implementation (check 3)
def np_layernorm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)                 # population variance, as nn.LayerNorm
    return (x - m) / np.sqrt(v + eps) * w + b


_erf = np.vectorize(math.erf)


def np_gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))          # exact GELU (activation="gelu")


def np_softmax(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


def np_encoder(x, model, heads):
    """Whole encoder in float64 from the module's state dict. Returns (Hn, [attention per layer])."""
    sd = {k: v.detach().double().cpu().numpy() for k, v in model.state_dict().items()}
    d = model.d_model
    dk = d // heads
    h = x @ sd["embed.weight"].T + sd["embed.bias"] + sd["pe"]
    atts = []
    n_layers = len(model.encoder.layers)
    for l in range(n_layers):
        p = f"encoder.layers.{l}."
        xn = np_layernorm(h, sd[p + "norm1.weight"], sd[p + "norm1.bias"])
        W, b = sd[p + "self_attn.in_proj_weight"], sd[p + "self_attn.in_proj_bias"]
        q, k, v = (xn @ W[i * d:(i + 1) * d].T + b[i * d:(i + 1) * d] for i in range(3))
        B, T, _ = q.shape
        split = lambda t: t.reshape(B, T, heads, dk).transpose(0, 2, 1, 3)           # (B, H, T, dk)
        q, k, v = split(q), split(k), split(v)
        a = np_softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(dk))                 # softmax(Q K^T / sqrt(d_k))
        atts.append(a)
        ctx = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, d)                         # concatenate the heads
        h = h + ctx @ sd[p + "self_attn.out_proj.weight"].T + sd[p + "self_attn.out_proj.bias"]
        xn = np_layernorm(h, sd[p + "norm2.weight"], sd[p + "norm2.bias"])
        ff = np_gelu(xn @ sd[p + "linear1.weight"].T + sd[p + "linear1.bias"]) @ sd[p + "linear2.weight"].T + sd[p + "linear2.bias"]
        h = h + ff
    return np_layernorm(h, sd["encoder.norm.weight"], sd["encoder.norm.bias"]), atts


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    say("# G report: Transformer temporal encoder (G1 module, G2 checks)")
    say()
    say("Nothing is trained. Design (decided 2026-10-29): embedding Linear(6, 64) + fixed sinusoidal PE; 2 x TransformerEncoderLayer(d_model 64, 4 heads, "
        "feed-forward 128, dropout 0.2, GELU, batch_first, pre-norm) + final LayerNorm(64); `enable_nested_tensor=False` (pre-norm layers do not support it); "
        "no attention mask (every position is a reading on or before the forecast time); output h_target = Hn[:, -1, :], the latest reading.")
    say()

    # ------------------------------------------------------------ data and module
    z = np.load(f"{ROOT}/preprocessing/c3_windows_gap.npz", allow_pickle=False)
    X, split, wid = z["X"], z["split"].astype(str), z["well_id"].astype(str)
    Wn, Lw, F = X.shape
    say("## G1. Module and data")
    say()
    say(f"- Windows: {Wn} (train {int((split == 'train').sum())}, val {int((split == 'val').sum())}, test {int((split == 'test').sum())}); "
        f"each window is {Lw} readings x {F} standardised features. X has {int(np.isnan(X).sum())} NaN.")
    torch.manual_seed(SEED)
    model = gt.TemporalEncoder(L=Lw, in_dim=F).eval()
    exp = gt.expected_parameter_counts(in_dim=F, d_model=model.d_model, ff=128, n_layers=len(model.encoder.layers))
    mod = {"embedding": sum(p.numel() for p in model.embed.parameters()),
           "layer": sum(p.numel() for p in model.encoder.layers[0].parameters()),
           "attention_per_layer": sum(p.numel() for p in model.encoder.layers[0].self_attn.parameters()),
           "feed_forward_per_layer": sum(p.numel() for n, p in model.encoder.layers[0].named_parameters() if n.startswith("linear")),
           "norms_per_layer": sum(p.numel() for n, p in model.encoder.layers[0].named_parameters() if n.startswith("norm")),
           "layers_total": sum(p.numel() for p in model.encoder.layers.parameters()),
           "final_norm": sum(p.numel() for p in model.encoder.norm.parameters()),
           "encoder": sum(p.numel() for p in model.encoder.parameters()),
           "total": model.n_parameters()}
    say()

    # batch: Keelvalur window 39 if it is a train window, otherwise the first Keelvalur train window
    kv = np.flatnonzero((wid == KEELVALUR) & (split == "train"))
    keel = int(kv[0])
    tr = [int(i) for i in np.flatnonzero(split == "train") if i != keel][:N_BATCH - 1]
    idx = [keel] + tr
    which = ("Keelvalur window 39 (a train window)" if keel == 39 else
             f"window 39 is not the first Keelvalur train window; the first Keelvalur train window is {keel}")
    say(f"- Check batch ({N_BATCH} train windows, eval mode unless stated): {idx}; first = {which}.")
    say()
    xb = torch.tensor(X[idx], dtype=torch.float32)
    RES["batch_windows"] = idx
    RES["keelvalur_window"] = keel

    say("## G2. Checks")
    say()

    # 1. shapes ------------------------------------------------------------------
    with torch.no_grad():
        h0 = model.embed(xb) + model.pe
        hn, ht = model(xb)
        with gt.no_fastpath():
            ref = model.encoder(h0)                              # PyTorch's own nn.TransformerEncoder forward (ordinary path)
    fast_before = torch.backends.mha.get_fastpath_enabled()
    with torch.no_grad():
        model(xb)
    check("the fused-kernel switch is restored after a forward pass", torch.backends.mha.get_fastpath_enabled() == fast_before)
    check(f"input {tuple(xb.shape)} -> H0 {tuple(h0.shape)} -> Hn {tuple(hn.shape)}, h_target {tuple(ht.shape)}",
          tuple(h0.shape) == (N_BATCH, Lw, model.d_model) and tuple(hn.shape) == (N_BATCH, Lw, model.d_model) and tuple(ht.shape) == (N_BATCH, model.d_model))
    check("no NaN in H0, Hn, h_target", not (torch.isnan(h0).any() or torch.isnan(hn).any() or torch.isnan(ht).any()).item())
    check("h_target equals Hn[:, -1, :]", torch.equal(ht, hn[:, -1, :]))
    dref = float((hn - ref).abs().max())
    check("module forward equals nn.TransformerEncoder's own forward", dref < 1e-6, f"max abs difference {dref:.2e}")

    # 2. parameter counts ---------------------------------------------------------
    rows = [[k, exp[k], mod[k], "PASS" if exp[k] == mod[k] else "FAIL"] for k in exp]
    ok_counts = all(exp[k] == mod[k] for k in exp)
    check("parameter counts from the formula equal the module's counts", ok_counts,
          f"embedding {mod['embedding']}, per layer {mod['layer']} = {mod['attention_per_layer']} + {mod['feed_forward_per_layer']} + {mod['norms_per_layer']}, "
          f"two layers {mod['layers_total']}, final LayerNorm {mod['final_norm']}, encoder {mod['encoder']}, total {mod['total']}")
    check("positional encoding is a buffer, not a parameter", "pe" in dict(model.named_buffers()) and "pe" not in dict(model.named_parameters())
          and not model.pe.requires_grad)
    say()
    for line in md_table(["part", "expected (formula)", "module", "status"], rows):
        say(line)
    say()
    RES["params"] = {"expected": exp, "module": mod}

    # 3. numpy hand check -----------------------------------------------------------
    heads = model.encoder.layers[0].self_attn.num_heads
    hn_np, att_np = np_encoder(xb.double().numpy(), model, heads)
    d_hand = float(np.abs(hn_np - hn.double().numpy()).max())
    check("module output equals the plain numpy encoder (embedding + PE, 2 pre-norm layers, final LayerNorm), eval mode", d_hand < 1e-5, f"max abs difference {d_hand:.2e}")
    RES["hand_check_max_abs_diff"] = d_hand

    # 4. attention weights ----------------------------------------------------------
    with torch.no_grad():
        _, _, att = model(xb, return_attention=True)
    dev = max(float((a.sum(-1) - 1).abs().max()) for a in att)
    neg = min(float(a.min()) for a in att)
    shp = [tuple(a.shape) for a in att]
    check("attention: every row sums to 1 and all weights are >= 0 (all layers, all heads)", dev < 1e-5 and neg >= 0,
          f"layers {len(att)}, shape per layer {shp[0]}, max |row sum - 1| = {dev:.2e}, min weight {neg:.2e}")
    d_att = max(float(np.abs(att[l].double().numpy() - att_np[l]).max()) for l in range(len(att)))
    check("module attention weights equal the numpy softmax(Q K^T / sqrt(d_k)) weights", d_att < 1e-5, f"max abs difference {d_att:.2e}")
    say()
    say(f"Layer 1 attention of the last position (position 8 = latest reading) over the 8 positions, {KEELVALUR} window {keel}, eval mode "
        "(information only: the weights are untrained, so they carry no meaning yet):")
    say()
    a1 = att[0][0, :, -1, :].numpy()
    for line in md_table(["head"] + [f"pos {p + 1}" for p in range(Lw)], [[f"head {h + 1}"] + [f"{v:.3f}" for v in a1[h]] for h in range(a1.shape[0])]):
        say(line)
    say()
    RES["attention_layer1_last_position_keelvalur"] = a1.tolist()
    RES["attention_row_sum_max_dev"] = dev

    # 5. why the positional encoding is needed ------------------------------------------
    w0 = xb[:1]
    rs = np.random.RandomState(SEED)
    perm = rs.permutation(Lw - 1)
    while (perm == np.arange(Lw - 1)).all():
        perm = rs.permutation(Lw - 1)
    order = np.concatenate([perm, [Lw - 1]])                       # shuffle the first 7 readings, keep the latest in place
    wp = w0[:, order, :]
    nope = copy.deepcopy(model).eval()
    nope.pe.zero_()                                                # copy without PE; the real module is untouched
    with torch.no_grad():
        a_diff = float((nope(w0)[1] - nope(wp)[1]).abs().max())
        b_diff = float((model(w0)[1] - model(wp)[1]).abs().max())
    check("without PE, permuting the first 7 readings leaves h_target unchanged (max diff < 1e-5)", a_diff < 1e-5, f"max difference {a_diff:.2e}; permutation {order.tolist()}")
    check("with PE, the same permutation changes h_target (max diff > 1e-3)", b_diff > 1e-3, f"max difference {b_diff:.3f}")
    RES["permutation"] = {"order": order.tolist(), "diff_without_pe": a_diff, "diff_with_pe": b_diff}

    # 6. dropout -----------------------------------------------------------------------
    model.train()
    with torch.no_grad():
        t1, t2 = model(xb)[1], model(xb)[1]
    model.eval()
    with torch.no_grad():
        e1, e2 = model(xb)[1], model(xb)[1]
    d_tr, d_ev = float((t1 - t2).abs().max()), float((e1 - e2).abs().max())
    check("train mode: two forward passes differ (dropout active)", d_tr > 0, f"max difference {d_tr:.3f}")
    check("eval mode: two forward passes are identical", d_ev == 0.0, f"max difference {d_ev}")

    # 7. gradients -----------------------------------------------------------------------
    model.train()
    model.zero_grad()
    model(xb)[1].sum().backward()
    names = [n for n, _ in model.named_parameters()]
    none = [n for n, p in model.named_parameters() if p.grad is None]
    nonfin = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    gmax_layers = max(float(p.grad.abs().max()) for n, p in model.named_parameters() if n.startswith("encoder.layers") and p.grad is not None)
    check("loss = h_target.sum(): every parameter has a gradient that is not None and finite", not none and not nonfin,
          f"{len(names)} parameter tensors; None: {len(none)}, non-finite: {len(nonfin)}")
    # h_target.sum() is almost blind inside the layers: the final LayerNorm output (weight 1, bias 0) sums to 0 for any input,
    # so its gradient w.r.t. the layers is ~0. A random projection of h_target gives a meaningful test.
    model.zero_grad()
    g = torch.Generator().manual_seed(SEED)
    r = torch.randn(model.d_model, generator=g)
    (model(xb)[1] @ r).sum().backward()
    zero_g = [n for n, p in model.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0.0]
    check("loss = h_target . r (fixed random vector): every parameter tensor has a non-zero gradient", not zero_g,
          f"tensors with all-zero gradient: {zero_g if zero_g else 'none'}")
    info(f"with loss = h_target.sum() the largest gradient inside the 2 encoder layers is {gmax_layers:.1e} (about 0): the sum of a LayerNorm output with weight 1 and bias 0 is 0 for any input, "
         "so that loss says nothing about the layers; the random-vector loss above is the informative gradient test.")
    model.zero_grad()
    model.eval()

    # 8. CPU vs CUDA ------------------------------------------------------------------------
    if torch.cuda.is_available():
        gname = torch.cuda.get_device_name(0)
        gpu = gt.TemporalEncoder(L=Lw, in_dim=F)
        gpu.load_state_dict(model.state_dict())
        gpu = gpu.cuda().eval()
        with torch.no_grad():
            hn_g, ht_g = gpu(xb.cuda())
        dg = max(float((hn_g.cpu() - hn).abs().max()), float((ht_g.cpu() - ht).abs().max()))
        check("CPU and CUDA outputs agree within 1e-4", dg < 1e-4, f"device {gname}, max abs difference {dg:.2e}")
        dg64 = float(np.abs(hn_g.cpu().double().numpy() - hn_np).max())
        check("CUDA output equals the float64 numpy encoder within 1e-5", dg64 < 1e-5, f"max abs difference {dg64:.2e}")
        # information: what PyTorch's own fused inference kernel (switched off inside the module) would give on this GPU
        with torch.no_grad():
            fused = gpu.encoder(gpu.embed(xb.cuda()) + gpu.pe).cpu()
        d_fused = float((fused - hn).abs().max())
        info(f"PyTorch's fused eval-mode CUDA kernel (not used by the module) differs from the CPU result by {d_fused:.2e}; "
             "the module switches it off so eval and train use the same arithmetic.")
        RES["cuda"] = {"device": gname, "max_abs_diff": dg, "max_abs_diff_vs_float64": dg64, "fused_kernel_diff_vs_cpu": d_fused}
    else:
        gpu = None
        check("CPU and CUDA outputs agree within 1e-4", None, "CUDA not available on this machine")
        RES["cuda"] = {"device": None}

    # 9. window length argument ------------------------------------------------------------------
    table = np.load(f"{ROOT}/encoding/pe_table.npy")
    ok_pe8 = table.shape == (8, 64) and float(np.abs(gt.positional_encoding(8, 64) - table).max()) < 1e-12 and torch.equal(model.pe, torch.tensor(table, dtype=torch.float32))
    check("PE(8, 64) from the formula equals encoding/pe_table.npy, and the module buffer is that table", ok_pe8,
          f"max abs difference {float(np.abs(gt.positional_encoding(8, 64) - table).max()):.1e}")
    for Lx in (4, 12):
        m = gt.TemporalEncoder(L=Lx).eval()
        torch.manual_seed(SEED)
        xr = torch.randn(4, Lx, F)
        with torch.no_grad():
            o, t_ = m(xr)
        check(f"window length L = {Lx}: Hn {tuple(o.shape)} and h_target {tuple(t_.shape)}", tuple(o.shape) == (4, Lx, 64) and tuple(t_.shape) == (4, 64) and not torch.isnan(o).any().item())
    check("PE for L = 4 equals the first 4 rows of the L = 8 table (positions do not depend on L)",
          float(np.abs(gt.positional_encoding(4, 64) - table[:4]).max()) < 1e-12)

    # 10. compatibility with Phase F -----------------------------------------------------------------
    import f_gcn                                                  # noqa: E402  (gnn/f_gcn.py, read only)
    nf = np.load(f"{ROOT}/gnn/f_node_features.npz", allow_pickle=False)
    same_windows = (nf["well_id"].astype(str) == wid).all() and (nf["last_input_date"].astype(str) == z["last_input_date"].astype(str)).all()
    torch.manual_seed(SEED)
    gcn = f_gcn.GCNEncoder(in_dim=nf["NF"].shape[2]).eval()
    with torch.no_grad():
        zz = gcn.target_embedding(gcn(torch.tensor(nf["NF"][idx], dtype=torch.float32)), nf["node_idx"][idx])
    check("Phase F windows are the same windows (well_id and last_input_date match)", bool(same_windows))
    check("h_target size equals the GCN z_target size (ready for the late fusion in Phase H)", ht.shape[-1] == zz.shape[-1] == gcn.out_dim,
          f"h_target {tuple(ht.shape)}, z_target {tuple(zz.shape)}")

    # 11. information: speed and memory ------------------------------------------------------------------
    say()
    xs = torch.tensor(X[np.flatnonzero(split == "train")[:128]], dtype=torch.float32)
    if gpu is not None:
        xs = xs.cuda()
        torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(5):
                gpu(xs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(20):
                gpu(xs)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 20 * 1000
        mem = torch.cuda.max_memory_allocated() / 2 ** 20
        info(f"forward pass of a batch of {xs.shape[0]} windows on {RES['cuda']['device']}: {ms:.2f} ms (mean of 20), peak GPU memory {mem:.1f} MiB (varies slightly between runs)")
        RES["speed"] = {"batch": int(xs.shape[0]), "ms_forward": ms, "peak_mib": mem}
    else:
        info("no CUDA device: speed and memory not measured")

    # ------------------------------------------------------------ summary
    fails = [n for n, s, _ in CHECKS if s == "FAIL"]
    skipped = [n for n, s, _ in CHECKS if s == "SKIPPED"]
    n_pass = sum(1 for _, s, _ in CHECKS if s == "PASS")
    say()
    say("## Files")
    say()
    say("- `transformer/g_transformer.py`, `transformer/run_phase_g.py`, `transformer/g_report.md`, `transformer/g_results.json`, `run_phase_g.bat`.")
    say()
    final = ("FAILED: " + "; ".join(fails)) if fails else ("ALL CHECKS PASSED" + (f" ({n_pass} passed, {len(skipped)} skipped: " + "; ".join(skipped) + ")" if skipped else ""))
    say(final)
    RES["final_line"] = final
    RES["checks"] = [{"name": n, "status": s, "detail": d} for n, s, d in CHECKS]
    with open(f"{HERE}/g_report.md", "w", newline="\n", encoding="utf8") as f:
        f.write("\n".join(L) + "\n")
    with open(f"{HERE}/g_results.json", "w", newline="\n") as f:
        json.dump(RES, f, indent=1, sort_keys=True)
        f.write("\n")
    print("\n".join(L))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
