"""
Phase F runner: builds the node features (F1), instantiates the GCN encoder (F2) and runs the encoder checks (F3).
Nothing is trained. Writes gnn/f_report.md and gnn/f_results.json, prints PASS/FAIL lines and ends with
ALL CHECKS PASSED or the list of failed checks.

    .\\run_phase_f.bat            (= .venv\\Scripts\\python.exe -B -I gnn\\run_phase_f.py)

numpy, csv and torch only (scipy is blocked on this PC). All random seeds = 42.
"""

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)                       # -I removes the script folder from sys.path, so add it explicitly
import f_node_features as f1                   # noqa: E402
import f_gcn                                   # noqa: E402

SEED = 42
KEELVALUR = "Keelvalur"
N_BATCH = 4

L, CHECKS, RES = [], [], {}


def say(s=""):
    L.append(s)


def md_table(header, rows):
    return f1.md_table(header, rows)


def check(name, ok, detail=""):
    if ok is None:
        CHECKS.append((name, "SKIPPED", detail))
        say(f"- SKIPPED: {name}" + (f" ({detail})" if detail else ""))
    else:
        CHECKS.append((name, "PASS" if ok else "FAIL", detail))
        say(f"- {'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    say("# F report: GCN spatial encoder (F1 node features, F2 encoder, F3 checks)")
    say()
    say("Nothing is trained. Node features use only readings dated on or before each window's last input date.")
    say()

    # ------------------------------------------------------------------ F1
    lines, res1, checks1 = f1.build()
    L.extend(lines)
    for n, ok, d in checks1:
        CHECKS.append((n, "PASS" if ok else "FAIL", d))
    RES["F1"] = res1

    nf = np.load(f"{ROOT}/gnn/f_node_features.npz", allow_pickle=False)
    NF, node_idx, wid, split = nf["NF"], nf["node_idx"], nf["well_id"].astype(str), nf["split"].astype(str)
    order = f_gcn.load_well_order()
    a_file = f_gcn.load_a_hat()                     # float64, as written by Phase D
    n_nodes = len(order)

    # ------------------------------------------------------------------ F2
    say()
    say("## F2. GCN encoder")
    say()
    torch.manual_seed(SEED)
    model = f_gcn.GCNEncoder(in_dim=NF.shape[2]).eval()
    say(f"- `GCNEncoder`: in_dim {model.in_dim}, hidden {model.hidden}, out {model.out_dim}, 2 layers, dropout 0.2, "
        "H1 = ReLU(A_hat (H0 W1) + b1), H2 = ReLU(A_hat (H1 W2) + b2); A_hat loaded from `graph/a_hat.npy` (not rebuilt), no extra identity; "
        "bias added after the A_hat product; Glorot initialisation, seed 42.")
    say()

    say("## F3. Encoder checks")
    say()
    tr_idx = np.flatnonzero(split == "train")[:N_BATCH]
    h0 = torch.tensor(NF[tr_idx], dtype=torch.float32)
    nidx = node_idx[tr_idx]

    # 1. shapes and NaN
    with torch.no_grad():
        Z = model(h0)
        zt = model.target_embedding(Z, nidx)
    check(f"Z shape is ({N_BATCH}, {n_nodes}, {model.out_dim}) for {N_BATCH} train windows", tuple(Z.shape) == (N_BATCH, n_nodes, model.out_dim), str(tuple(Z.shape)))
    check(f"z_target shape is ({N_BATCH}, {model.out_dim})", tuple(zt.shape) == (N_BATCH, model.out_dim), str(tuple(zt.shape)))
    check("no NaN in Z", not torch.isnan(Z).any().item())
    check("z_target equals Z[b, node_idx[b], :]", all(torch.equal(zt[b], Z[b, int(nidx[b])]) for b in range(N_BATCH)))

    # 2. parameter count
    expect = model.in_dim * model.hidden + model.hidden + model.hidden * model.out_dim + model.out_dim
    npar = model.n_parameters()
    check(f"parameter count = {model.in_dim}*{model.hidden} + {model.hidden} + {model.hidden}*{model.out_dim} + {model.out_dim}", npar == expect, f"{npar} (expected {expect})")
    RES["params"] = int(npar)

    # 3. numpy hand check, eval mode
    A = model.a_hat.double().numpy()
    W1, b1 = model.W1.detach().double().numpy(), model.b1.detach().double().numpy()
    W2, b2 = model.W2.detach().double().numpy(), model.b2.detach().double().numpy()
    H = h0.double().numpy()
    H1 = np.maximum(A @ (H @ W1) + b1, 0.0)
    H2 = np.maximum(A @ (H1 @ W2) + b2, 0.0)
    diff = float(np.abs(H2 - Z.double().numpy()).max())
    check("module output equals the plain numpy two-layer calculation (eval mode)", diff < 1e-5, f"max abs difference {diff:.2e}")
    RES["hand_check_max_abs_diff"] = diff

    # 4. A_hat in the module
    a32 = a_file.astype(np.float32)
    exact = bool(np.array_equal(model.a_hat.numpy(), a32))
    d64 = float(np.abs(model.a_hat.double().numpy() - a_file).max())
    asym = float(np.abs(a_file - a_file.T).max())
    rs = a_file.sum(axis=1)
    check("A_hat in the module equals graph/a_hat.npy exactly (float32 copy)", exact, f"max difference to the float64 file {d64:.2e} (float32 rounding)")
    check("A_hat symmetric", asym < 1e-12, f"max asymmetry {asym:.2e}")
    check("A_hat row sums within [0.80, 1.10]", rs.min() >= 0.80 and rs.max() <= 1.10, f"min {rs.min():.4f}, max {rs.max():.4f}")
    RES["a_hat"] = {"max_asymmetry": asym, "row_sum_min": float(rs.min()), "row_sum_max": float(rs.max())}

    # 5. neighbour influence for Keelvalur
    i = order.index(KEELVALUR)
    others = [j for j in range(n_nodes) if j != i]
    j_near = max(others, key=lambda j: a_file[i, j])
    j_far = min(others, key=lambda j: a_file[i, j])
    cand = np.flatnonzero((wid == KEELVALUR) & (split == "train"))
    wk = int(cand[0] if cand.size else np.flatnonzero(wid == KEELVALUR)[0])
    base = torch.tensor(NF[wk:wk + 1], dtype=torch.float32)

    def target_vec(h):
        with torch.no_grad():
            return model.target_embedding(model(h), [i])[0]

    z0 = target_vec(base)

    def change(j, feat):
        h = base.clone()
        h[0, j, feat] += 1.0
        return float((target_vec(h) - z0).norm())

    near0, far0 = change(j_near, 0), change(j_far, 0)
    allf = [(change(j_near, f), change(j_far, f)) for f in range(model.in_dim)]
    say()
    check("adding 1.0 to feature 0 of the nearest neighbour changes z_target more than adding it to the farthest well", near0 > far0,
          f"{KEELVALUR} window {wk}; nearest {order[j_near]} (A_hat {a_file[i, j_near]:.4f}): change {near0:.4f}; "
          f"farthest {order[j_far]} (A_hat {a_file[i, j_far]:.2e}): change {far0:.2e}")
    say()
    say("Change in z_target (L2 norm) when +1.0 is added to one input feature of one node, same window, eval mode (information; the check above uses feature 0):")
    say()
    fn = ["feature %d" % f for f in range(model.in_dim)]
    for line in md_table(["node"] + fn, [[f"nearest: {order[j_near]}"] + [f"{a:.4f}" for a, _ in allf],
                                          [f"farthest: {order[j_far]}"] + [f"{b:.2e}" for _, b in allf]]):
        say(line)
    say()
    RES["influence"] = {"well": KEELVALUR, "window": wk, "nearest": order[j_near], "nearest_weight": float(a_file[i, j_near]),
                        "farthest": order[j_far], "farthest_weight": float(a_file[i, j_far]), "change_nearest": near0, "change_farthest": far0}

    # 6. effective reach of 2 layers (information)
    A2 = a_file @ a_file
    selfw = np.diag(A2)
    rowsum = A2.sum(axis=1)
    outside = (rowsum - selfw) / rowsum
    n_gt50 = int((outside > 0.5).sum())
    o = np.argsort(selfw, kind="stable")
    say("Effective reach of 2 layers, A2 = A_hat @ A_hat (information for the paper, not an assert). Self weight A2[i,i] over the 53 wells:")
    say()
    for line in md_table(["statistic", "value"], [["min", f"{selfw.min():.4f} ({order[o[0]]})"], ["median", f"{np.median(selfw):.4f}"],
                                                  ["max", f"{selfw.max():.4f} ({order[o[-1]]})"],
                                                  ["wells with > 50% of their A2 row weight outside themselves", f"{n_gt50} of {n_nodes}"]]):
        say(line)
    say()
    say("Five smallest self weights: " + ", ".join(f"{order[k]} {selfw[k]:.4f}" for k in o[:5]) + ".")
    say("Five largest self weights: " + ", ".join(f"{order[k]} {selfw[k]:.4f}" for k in o[::-1][:5]) + ".")
    say()
    RES["A2_self_weight"] = {"min": float(selfw.min()), "min_well": order[o[0]], "median": float(np.median(selfw)),
                             "max": float(selfw.max()), "max_well": order[o[-1]], "wells_over_50pct_outside": n_gt50}
    check("A2 self weights computed for every well (information)", len(selfw) == n_nodes and np.isfinite(selfw).all(), f"{n_gt50} of {n_nodes} wells have > 50% outside weight")

    # 7. dropout
    model.train()
    with torch.no_grad():
        t1, t2 = model(h0), model(h0)
    train_diff = float((t1 - t2).abs().max())
    model.eval()
    with torch.no_grad():
        e1, e2 = model(h0), model(h0)
    eval_diff = float((e1 - e2).abs().max())
    check("train mode: two forward passes differ (dropout active)", train_diff > 0, f"max difference {train_diff:.3f}")
    check("eval mode: two forward passes are identical", eval_diff == 0.0, f"max difference {eval_diff}")
    RES["dropout"] = {"train_mode_diff": train_diff, "eval_mode_diff": eval_diff}

    # 8. CPU and CUDA
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        gpu = f_gcn.GCNEncoder(in_dim=NF.shape[2]).to("cuda")
        gpu.load_state_dict(model.state_dict())
        gpu.eval()
        with torch.no_grad():
            zg = gpu(h0.to("cuda")).cpu()
        dev = float((zg - Z).abs().max())
        check("CPU and CUDA outputs agree within 1e-4", dev < 1e-4, f"device {name}, max abs difference {dev:.2e}")
        RES["cuda"] = {"device": name, "max_abs_diff": dev}
    else:
        check("CPU and CUDA outputs agree within 1e-4", None, "CUDA not available on this machine")
        RES["cuda"] = {"device": None}

    # ------------------------------------------------------------------ summary
    fails = [n for n, s, _ in CHECKS if s == "FAIL"]
    skipped = [n for n, s, _ in CHECKS if s == "SKIPPED"]
    n_pass = sum(1 for _, s, _ in CHECKS if s == "PASS")
    say()
    say("## Files")
    say()
    say("- `gnn/f_node_features.py`, `gnn/f_node_features.npz`, `gnn/f_staleness_scaler.json`, `gnn/f_gcn.py`, `gnn/run_phase_f.py`, `gnn/f_report.md`, `gnn/f_results.json`, `run_phase_f.bat`.")
    say()
    if fails:
        final = "FAILED: " + "; ".join(fails)
    elif skipped:
        final = f"ALL CHECKS PASSED ({n_pass} passed, {len(skipped)} skipped: " + "; ".join(skipped) + ")"
    else:
        final = "ALL CHECKS PASSED"
    say(final)
    RES["final_line"] = final
    RES["checks"] = [{"name": n, "status": s, "detail": d} for n, s, d in CHECKS]
    with open(f"{HERE}/f_report.md", "w", newline="\n", encoding="utf8") as f:
        f.write("\n".join(L) + "\n")
    with open(f"{HERE}/f_results.json", "w", newline="\n") as f:
        json.dump(RES, f, indent=1, sort_keys=True)
        f.write("\n")
    print("\n".join(L))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
