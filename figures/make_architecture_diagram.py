"""Detailed architecture figure: data layer to prediction. Real mini-visuals from project files.
Usage: python3 make_arch.py <project_root> <out_prefix> [progress]
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

ROOT = sys.argv[1]
OUT = sys.argv[2]
PROGRESS = len(sys.argv) > 3 and sys.argv[3] == "progress"

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
W, H = 130, 90
fig = plt.figure(figsize=(13, 9 * 113 / 90), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(-23, H); ax.axis("off")

INK = "#1F2933"; MUTE = "#5B6670"; LINE = "#8A939B"
BLUE, BLUE_BG = "#2F5F98", "#EAF1F9"
GREEN, GREEN_BG = "#1B7F5B", "#E7F4EE"
ORANGE, ORANGE_BG = "#B4601A", "#FDF1E3"
PURPLE, PURPLE_BG = "#5E4B9A", "#EEEAF7"
GREY_BG = "#F4F5F7"; GREY_EC = "#9AA3AB"
DIST = {"Thanjavur": "#E69F00", "Thiruvarur": "#0072B2", "Nagapattinam": "#009E73"}


def rbox(x, y, w, h, fc, ec, lw=1.2, r=0.9, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p); return p


def txt(x, y, s, size=10.5, color=INK, weight="normal", ha="center", va="center", style="normal", z=5, rot=0):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, style=style, zorder=z, rotation=rot)


def arrow(x1, y1, x2, y2, color=LINE, lw=1.6, ms=12, z=4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0))


def line(xs, ys, color=LINE, lw=1.6, z=4):
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="butt")


def mark(label, x, y, built):
    if not PROGRESS:
        return
    ax.add_patch(Circle((x, y), 1.55, fc="#1B7F5B" if built else "#8A8F98", ec="white", lw=1.5, zorder=9))
    txt(x, y, label, size=9, color="white", weight="bold", z=10)


def vec(x, y, n, cm, ec, lo, step, cw=2.0, ch=1.9, horizontal=False):
    for t in range(n):
        xx, yy = (x + t * cw, y) if horizontal else (x, y + t * ch)
        ax.add_patch(Rectangle((xx, yy), cw, ch, fc=cm(lo + step * t), ec=ec, lw=0.7, zorder=6))


# ---------------------------------------------------------------- real data
z = np.load(f"{ROOT}/preprocessing/c3_windows_gap.npz")
wells = json.load(open(f"{ROOT}/graph/well_order.json"))["wells"]
pe = np.load(f"{ROOT}/encoding/pe_table.npy")
A = np.load(f"{ROOT}/graph/a_hat.npy")
sc = json.load(open(f"{ROOT}/preprocessing/c2_scaler.json"))["features"]["depth_m"]
k = z["well_id"].tolist().index("Keelvalur")
Xw = z["X"][k]
depth_raw = Xw[:, 5] * sc["std_used"] + sc["mean"]
target = float(z["y_metres"][k])

# ---------------------------------------------------------------- column headers
for x, s, c in [(12, "1  DATA LAYER", BLUE), (39, "2  PREPARATION", MUTE), (79.5, "3  TWO-BRANCH ENCODER", INK), (118, "4  PREDICTION", PURPLE)]:
    txt(x, 87.4, s, size=12.5, color=c, weight="bold")
line([1, 129], [85.2, 85.2], color="#C9CED3", lw=1.0)

# ================================================================ 1 DATA LAYER
dx, dw = 1.5, 21
cards = [
    ("CGWB groundwater", "depth to water level (m)\n53 wells, 1,232 readings\nquarterly, 2013 to 2024", 70),
    ("IMD rainfall", "0.25° grid, 18 cells\ndaily, 2013 to 2025", 56),
    ("IMD temperature", "1° grid, 9 cells\ndaily Tmax and Tmin", 42),
    ("NASA POWER", "relative humidity and\nevapotranspiration\npoint value at each well", 28),
]
for t, b, y in cards:
    rbox(dx, y, dw, 12.5, BLUE_BG, BLUE, lw=1.3)
    rbox(dx, y, 1.3, 12.5, BLUE, BLUE, lw=0, r=0.4)
    txt(dx + dw / 2 + 0.6, y + 10.2, t, size=10.8, color=BLUE, weight="bold")
    txt(dx + dw / 2 + 0.6, y + 4.7, b, size=9.3, color=INK)
txt(dx + 4.2, 24.0, "target", size=9, color=BLUE, weight="bold", ha="left")
txt(dx + 9.8, 24.0, "= depth (CGWB);", size=9, color=MUTE, ha="left")
txt(dx + 4.2, 21.3, "inputs", size=9, color=BLUE, weight="bold", ha="left")
txt(dx + 9.8, 21.3, "= weather and depth", size=9, color=MUTE, ha="left")
mark("A", dx + dw - 0.4, 83.0, True)

# ================================================================ 2 PREPARATION
px, pw = 27, 24
rbox(px, 66, pw, 17, GREY_BG, GREY_EC)
txt(px + pw / 2, 80.3, "Integration", size=10.8, weight="bold")
txt(px + pw / 2, 72.4, "nearest grid cell per well\nmeans and sums before each\nreading: rain 180 d; Tmax,\nTmin, RH 90 d mean; ET 90 d sum", size=8.9)
mark("B", px + pw - 0.4, 83.0, True)

rbox(px, 50, pw, 13.5, GREY_BG, GREY_EC)
txt(px + pw / 2, 60.6, "Integrated table", size=10.8, weight="bold")
txt(px + pw / 2, 54.6, "1,232 readings, 53 wells\n5 weather inputs + depth", size=9.2)
mark("B", px + pw - 0.4, 63.5, True)

rbox(px, 31, pw, 16.5, GREY_BG, GREY_EC)
txt(px + pw / 2, 44.6, "Preparation", size=10.8, weight="bold")
txt(px + pw / 2, 37.4, "1,211 readings after duplicates\ntarget = next reading\nz-score fitted on training\nperiod only (Eq. 7)", size=9.0)
mark("C", px + pw - 0.4, 47.5, True)

rbox(px, 9.2, pw, 18.8, GREY_BG, GREY_EC)
txt(px + pw / 2, 25.1, "Sliding windows, L = 8", size=10.8, weight="bold")
txt(px + pw / 2, 21.0, "840 windows, 42 wells", size=9.2)
txt(px + pw / 2, 17.9, "chronological split by target date", size=8.3, color=MUTE, style="italic")
bx0, bw = px + 1.6, pw - 3.2
for f0, f1, col in [(0, 581 / 840, "#0072B2"), (581 / 840, 728 / 840, "#E69F00"), (728 / 840, 1, "#CC79A7")]:
    ax.add_patch(Rectangle((bx0 + f0 * bw, 13.4), (f1 - f0) * bw, 2.4, fc=col, ec="white", lw=0.8, zorder=5))
txt(bx0 + 0.40 * bw, 11.4, "train 581", size=8.4, color="#0072B2", weight="bold")
txt(bx0 + 0.82 * bw, 11.4, "147", size=8.4, color="#B87A00", weight="bold")
txt(bx0 + 0.97 * bw, 11.4, "112", size=8.4, color="#A14F82", weight="bold")
mark("C", px + pw - 0.4, 28.0, True)

# data cards -> integration
for y in (76.2, 62.2, 48.2, 34.2):
    line([dx + dw, 24.9], [y, y], lw=1.2)
line([24.9, 24.9], [34.2, 76.2], lw=1.2)
arrow(24.9, 74.5, px, 74.5)
for ya, yb in [(66, 63.7), (50, 47.7), (31, 28.2)]:
    arrow(px + pw / 2, ya, px + pw / 2, yb)

# ================================================================ 3 BRANCHES
bx, bwid = 55, 49
# ---- spatial
rbox(bx, 47, bwid, 36.5, GREEN_BG, GREEN, lw=1.6, r=1.2)
txt(bx + 1.5, 81.3, "Spatial branch: graph over the wells", size=11.6, color=GREEN, weight="bold", ha="left")
gx = [bx + 2, bx + 18.5, bx + 35]
labels = [("Distance", "haversine, 53 × 53\n0.155 to 125.3 km"),
          ("Edge weight", r"$A_{ij}=e^{-d_{ij}^{2}/\sigma^{2}}$" + "\n" + r"$\sigma$ = 24.8 km"),
          ("Normalise", r"$\hat{A}=D^{-1/2}AD^{-1/2}$" + "\nrows sum to about 1")]
for xx, (t, b) in zip(gx, labels):
    rbox(xx, 69.3, 13.4, 9.8, "white", GREEN, lw=1.2, r=0.7)
    txt(xx + 6.7, 77.1, t, size=10, color=GREEN, weight="bold")
    txt(xx + 6.7, 72.2, b, size=8.6, color=INK)
arrow(gx[0] + 13.4, 74.2, gx[1], 74.2, color=GREEN, lw=1.4)
arrow(gx[1] + 13.4, 74.2, gx[2], 74.2, color=GREEN, lw=1.4)
mark("D", bx + bwid - 1.8, 81.6, True)

lat = np.array([w["lat"] for w in wells]); lon = np.array([w["lon"] for w in wells])
mx0, my0, mw, mh = bx + 2, 49.5, 13.5, 15.5
rbox(mx0, my0, mw, mh, "white", "#B9D8C9", lw=1.0, r=0.6)
sx = (lon - lon.min()) / (lon.max() - lon.min()); sy = (lat - lat.min()) / (lat.max() - lat.min())
mxs = mx0 + 1.2 + sx * (mw - 2.4); mys = my0 + 3.6 + sy * (mh - 5.2)
for i in range(53):
    for j in range(i + 1, 53):
        if A[i, j] > 0.075:
            ax.plot([mxs[i], mxs[j]], [mys[i], mys[j]], color="#9DBFAE", lw=0.55, zorder=3)
for i, w in enumerate(wells):
    ax.add_patch(Circle((mxs[i], mys[i]), 0.5, fc=DIST[w["district"]], ec="white", lw=0.3, zorder=5))
txt(mx0 + mw / 2, my0 + 1.5, "53 wells = nodes", size=8.6, color=GREEN, weight="bold")

ys = np.linspace(52.6, 62.4, 5)
c1, c2, c3 = bx + 25.5, bx + 33.5, bx + 41.5
for ya in ys:
    for yb in ys:
        ax.plot([c1, c2], [ya, yb], color="#A9CDBB", lw=0.5, zorder=3)
        ax.plot([c2, c3], [ya, yb], color="#A9CDBB", lw=0.5, zorder=3)
for cx_, fc in ((c1, "white"), (c2, "#C3E3D3"), (c3, "#7CC3A0")):
    for y in ys:
        ax.add_patch(Circle((cx_, y), 0.95, fc=fc, ec=GREEN, lw=1.1, zorder=6))
txt((c1 + c3) / 2, 67.5, "GCN, two layers", size=9.4, color=GREEN, weight="bold")
txt(c2, 65.4, "layer 1", size=8.2, color=GREEN)
txt(c3, 65.4, "layer 2", size=8.2, color=GREEN)
txt((c1 + c2) / 2 + 1.5, 50.0, r"$H^{(1)}=\sigma(\hat{A}H^{(0)}W^{(0)})$", size=8.8)
arrow(mx0 + mw, 57.2, c1 - 1.8, 57.2, color=GREEN, lw=1.4)
txt(mx0 + mw + 2.0, 59.0, "X", size=9.6, color=GREEN, weight="bold", style="italic")
hx = bx + 45.3
vec(hx, 51.2, 6, plt.cm.Greens, GREEN, 0.35, 0.1, cw=2.0, ch=2.1)
txt(hx + 1.0, 49.2, r"$h_{s}$", size=10, color=GREEN)
arrow(c3 + 1.2, 57.2, hx - 0.4, 57.2, color=GREEN, lw=1.4)
mark("F", c3 + 3.4, 67.5, False)

# ---- temporal
rbox(bx, 9.5, bwid, 34.5, ORANGE_BG, ORANGE, lw=1.6, r=1.2)
txt(bx + 1.5, 41.9, "Temporal branch: Transformer encoder", size=11.6, color=ORANGE, weight="bold", ha="left")
wx0, wy0, cw, ch = bx + 2.5, 29.4, 1.35, 1.35
for q in range(8):
    for f in range(6):
        v = np.clip(Xw[q, f], -2, 2)
        ax.add_patch(Rectangle((wx0 + q * cw, wy0 + (5 - f) * ch), cw, ch, fc=plt.cm.Blues(0.18 + 0.62 * (v + 2) / 4), ec="white", lw=0.5, zorder=6))
txt(wx0 + 4 * cw, 39.2, r"window $X_i$", size=9.6, weight="bold")
txt(wx0 + 4 * cw, 27.0, "8 × 6, Keelvalur example", size=8.0, color=MUTE, style="italic")
mark("E", bx + bwid - 1.8, 41.9, True)
ex = wx0 + 8 * cw + 1.0
rbox(ex + 2.4, 30.4, 10.6, 8.6, "white", ORANGE, lw=1.1, r=0.7)
txt(ex + 7.7, 37.2, "Embed", size=10, color=ORANGE, weight="bold")
txt(ex + 7.7, 33.2, r"$XW_e+b_e$" + "\n6 → 64", size=8.8, color=MUTE)
arrow(ex, 34.7, ex + 2.4, 34.7, color=ORANGE, lw=1.4)
plus_x = ex + 15.0
txt(plus_x, 34.7, "+", size=15, color=ORANGE, weight="bold")
pex, pey, pw_, ph_ = plus_x + 1.8, 29.8, 10.6, 9.0
ax.imshow(pe[:, ::4], extent=(pex, pex + pw_, pey, pey + ph_), cmap="BrBG", vmin=-1.3, vmax=1.3, aspect="auto", zorder=6, interpolation="nearest")
ax.add_patch(Rectangle((pex, pey), pw_, ph_, fc="none", ec=ORANGE, lw=1.1, zorder=7))
txt(pex + pw_ / 2, 40.4, "sinusoidal PE", size=9.2, color=ORANGE, weight="bold")
txt(pex + pw_ / 2, 28.3, "8 × 64, order of quarters", size=8.0, color=MUTE)
arrow(ex + 13.0, 34.7, plus_x - 1.2, 34.7, color=ORANGE, lw=1.4)
arrow(pex + pw_ / 2, 27.5, pex + pw_ / 2, 24.8, color=ORANGE, lw=1.4)

tx0, ty0, tw, th = bx + 2.5, 10.6, 33.5, 14.0
rbox(tx0, ty0, tw, th, "white", ORANGE, lw=1.3, r=0.8)
txt(tx0 + 1.6, ty0 + th - 1.6, "×2 layers", size=9.0, color=ORANGE, weight="bold", ha="left")
txt(tx0 + tw - 1.4, ty0 + th - 1.6, "dropout 0.2", size=8.0, color=MUTE, style="italic", ha="right")
rng = np.random.default_rng(3)
att = rng.random((8, 8)) ** 2.2
ax.imshow(att, extent=(tx0 + 1.8, tx0 + 8.0, ty0 + 3.4, ty0 + 9.6), cmap="Oranges", aspect="auto", zorder=6, interpolation="nearest")
ax.add_patch(Rectangle((tx0 + 1.8, ty0 + 3.4), 6.2, 6.2, fc="none", ec=ORANGE, lw=0.9, zorder=7))
txt(tx0 + 4.9, ty0 + 1.0, "attention\n(illustrative)", size=6.4, color=MUTE, style="italic")
for i, lab in enumerate(["Multi-head self-attention, 4 heads", "Add & Norm", "Feed-forward, Add & Norm"]):
    yy = ty0 + 7.4 - i * 3.0
    rbox(tx0 + 10.0, yy - 1.2, 22.2, 2.5, ORANGE_BG, ORANGE, lw=0.9, r=0.4)
    txt(tx0 + 21.1, yy + 0.05, lab, size=8.5, color=INK)
mark("G", tx0 + tw + 1.6, ty0 + th + 0.4, False)
htx = bx + 41.5
vec(htx, 13.2, 6, plt.cm.Oranges, ORANGE, 0.30, 0.11, cw=2.0, ch=2.1)
txt(htx + 1.0, 11.0, r"$h_{t}$", size=10, color=ORANGE)
arrow(tx0 + tw, 17.8, htx - 0.4, 17.8, color=ORANGE, lw=1.4)

# prep -> branches (vertical lines in the gap x = 53)
line([px + pw, 53.0], [56.7, 56.7], lw=1.6); line([53.0, 53.0], [56.7, 74.2], lw=1.6); arrow(53.0, 74.2, gx[0], 74.2, lw=1.6)
txt(53.0, 65.0, "well coordinates", size=7.8, color=MUTE, style="italic", rot=90, z=6).set_bbox(dict(fc="white", ec="none", pad=0.6))
line([px + pw, 53.0], [18.6, 18.6], lw=1.6); line([53.0, 53.0], [18.6, 34.7], lw=1.6); arrow(53.0, 34.7, bx, 34.7, lw=1.6)
txt(53.0, 26.5, "windows", size=7.8, color=MUTE, style="italic", rot=90, z=6).set_bbox(dict(fc="white", ec="none", pad=0.6))

# ================================================================ 4 PREDICTION
fx, fw = 107, 22
rbox(fx, 61, fw, 22, PURPLE_BG, PURPLE, lw=1.6, r=1.2)
txt(fx + fw / 2, 80.4, "Late fusion", size=11.2, color=PURPLE, weight="bold")
txt(fx + fw / 2, 77.6, "branches stay separate until here", size=7.6, color=MUTE, style="italic")
vx = fx + 6.5
vec(vx, 73.2, 6, plt.cm.Greens, GREEN, 0.35, 0.11, cw=2.0, ch=2.0, horizontal=True)
vec(vx, 70.4, 6, plt.cm.Oranges, ORANGE, 0.30, 0.11, cw=2.0, ch=2.0, horizontal=True)
txt(vx - 3.2, 74.2, r"$h_s$", size=9, color=GREEN); txt(vx - 3.2, 71.4, r"$h_t$", size=9, color=ORANGE)
txt(fx + fw / 2, 65.4, r"$u=W_f[h_s;h_t]+b_f$", size=9.8)
mark("H", fx + fw - 1.6, 83.0, False)

rbox(fx, 41, fw, 17.5, PURPLE_BG, PURPLE, lw=1.6, r=1.2)
txt(fx + fw / 2, 55.4, "Prediction head", size=11.2, color=PURPLE, weight="bold")
txt(fx + fw / 2, 49.4, r"$\hat{y}_{i}=W_p\,u_i+b_p$", size=10.4)
txt(fx + fw / 2, 44.0, "next-reading depth, m", size=8.8, color=MUTE)
mark("I", fx + fw - 1.6, 58.5, False)

rbox(fx, 20.5, fw, 17.5, "white", PURPLE, lw=1.4, r=1.0)
txt(fx + fw / 2, 35.4, "Output and target", size=10.6, color=PURPLE, weight="bold")
gx0, gy0, gw, gh = fx + 3.6, 24.2, fw - 6.8, 8.2
lo, hi = 1.2, 5.6
def ypos(v): return gy0 + (hi - v) / (hi - lo) * gh
xs = gx0 + np.arange(9) / 8 * gw
ax.plot(xs[:8], [ypos(v) for v in depth_raw], color="#355C7D", lw=1.3, zorder=6, marker="o", ms=2.6)
ax.plot([xs[7], xs[8]], [ypos(depth_raw[7]), ypos(target)], color=PURPLE, lw=1.2, ls="--", zorder=6)
ax.plot([xs[8]], [ypos(target)], marker="*", color=PURPLE, ms=9, zorder=7)
line([gx0 - 1, gx0 - 1], [gy0 - 0.3, gy0 + gh + 0.3], color=LINE, lw=0.9)
txt(gx0 - 2.0, gy0 + gh / 2, "depth", size=7.4, color=MUTE, rot=90)
txt(fx + fw / 2, 22.0, "Keelvalur: 8 readings, then target", size=7.4, color=MUTE, style="italic")

rbox(fx, 9.5, fw, 8.2, PURPLE_BG, PURPLE, lw=1.2, r=0.8)
txt(fx + fw / 2, 13.6, r"Loss = MSE + $\lambda\,\|\Theta\|_2^2$", size=9.8)
mark("I", fx + fw - 1.6, 17.7, False)

# branch outputs -> fusion
line([hx + 2.4, 105.0], [57.2, 57.2], color=GREEN, lw=1.6)
line([105.0, 105.0], [57.2, 74.2], color=GREEN, lw=1.6)
arrow(105.0, 74.2, vx - 0.2, 74.2, color=GREEN, lw=1.6)
line([htx + 2.4, 106.0], [17.8, 17.8], color=ORANGE, lw=1.6)
line([106.0, 106.0], [17.8, 71.4], color=ORANGE, lw=1.6)
arrow(106.0, 71.4, vx - 0.2, 71.4, color=ORANGE, lw=1.6)
arrow(fx + fw / 2, 61, fx + fw / 2, 58.7, color=PURPLE, lw=1.6)
arrow(fx + fw / 2, 41, fx + fw / 2, 38.2, color=PURPLE, lw=1.6)
arrow(fx + fw / 2, 20.5, fx + fw / 2, 17.9, color=PURPLE, lw=1.6)

# ================================================================ experiments and output
line([1, 129], [5.6, 5.6], color="#C9CED3", lw=1.0)
txt(65, 2.4, "5  TRAINING, COMPARISON AND OUTPUT", size=12.5, color=INK, weight="bold")
blocks = [
 ("J", "Training setup", "how the model is fitted", ["Adam, learning rate 1e-4", "batch 16 to 32", "100 to 200 epochs", "early stopping", "dropout 0.2, L2 penalty", "RTX 3050, 6 GB"], PURPLE),
 ("K", "Baselines", "does the complexity pay off?", ["persistence (last reading)", "ARIMA / SARIMA", "per-well LSTM", "per-well CNN", "GNN + LSTM, no attention", "Transformer only (optional)"], BLUE),
 ("L", "Ablations", "which part gives the gain?", ["full model", "no GNN (Transformer only)", "no Transformer (GNN + LSTM)", "kernel width 10, 25, 49 km"], GREEN),
 ("M", "Robustness", "does it hold outside clean data?", ["10/30/50% readings dropped", "wells unseen in training", "recent unseen years", "strict windows, 157 / 23"], ORANGE),
 ("N", "Metrics", "how good, in metres", ["MAE in metres", "RMSE and R²", "per well and averaged", "physical plausibility check", "spatial coherence check"], PURPLE),
 ("O", "Output", "what the user receives", ["per-well forecast", "district average (3 districts)", "risk class (CGWB names):", "Safe to Over-exploited", "maps and plots"], BLUE),
]
bw_, gap_, x_ = 20.3, 1.04, 1.5
for lab, t, why, items, col in blocks:
    rbox(x_, -22.5, bw_, 21.0, "white", col, lw=1.3, r=0.8)
    rbox(x_, -3.8, bw_, 2.3, col, col, lw=0, r=0.5)
    txt(x_ + 1.0, -2.65, t, size=10.2, color="white", weight="bold", ha="left")
    txt(x_ + bw_ / 2, -6.0, why, size=8.2, color=MUTE, style="italic")
    for i, it in enumerate(items):
        txt(x_ + 1.0, -9.2 - i * 2.2, "\u2022 " + it, size=8.1, color=INK, ha="left")
    mark(lab, x_ + bw_ - 1.0, -1.5, False)
    x_ += bw_ + gap_
if PROGRESS:
    ax.add_patch(Circle((3.5, 17.0), 1.1, fc="#1B7F5B", ec="white", zorder=9)); txt(5.3, 17.0, "built and checked", size=8.6, ha="left")
    ax.add_patch(Circle((3.5, 13.5), 1.1, fc="#8A8F98", ec="white", zorder=9)); txt(5.3, 13.5, "designed, not built yet", size=8.6, ha="left")

fig.savefig(OUT + ".png", dpi=200)
fig.savefig(OUT + ".pdf")
print("saved", OUT)
