import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

import os
CSV = "data/cgwb_thanjavur_thiruvarur_nagapattinam_2013_2025.csv"  # run from the project root
if not os.path.exists(CSV):
    CSV = "/mnt/user-data/uploads/groundwater-project/data/cgwb_thanjavur_thiruvarur_nagapattinam_2013_2025.csv"
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
INK = "#1f2933"
GREEN, GREEN_BG = "#2f7d5b", "#e8f5ee"
ORANGE, ORANGE_BG = "#b86a1d", "#fdf2e4"
PURPLE, PURPLE_BG = "#6b4c9a", "#f0eaf8"
BLUE, BLUE_BG = "#4a6785", "#edf2f8"
DIST = {"Thanjavur": "#e69f00", "Thiruvarur": "#0072b2", "Nagapattinam": "#009e73"}

fig, ax = plt.subplots(figsize=(15, 8.2))
ax.set_xlim(0, 150)
ax.set_ylim(0, 82)
ax.axis("off")


def rbox(x, y, w, h, fc, ec, lw=1.4, r=1.4, z=1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))


def txt(x, y, s, size=9.5, bold=False, color=INK, ha="center", va="center", z=5, style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight="bold" if bold else "normal", color=color, ha=ha, va=va, zorder=z, style=style)


def arr(p0, p1, color=INK, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=lw, color=color, ls=ls, zorder=6, shrinkA=0, shrinkB=0))


def line(pts, color=INK, lw=1.6, head=True):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] if head else xs, ys[:-1] if head else ys, color=color, lw=lw, zorder=6, solid_capstyle="butt")
    if head:
        arr(pts[-2], pts[-1], color=color, lw=lw)


def bar(x, y, w, h, color, n=6, alpha_lo=0.35):
    for i in range(n):
        a = alpha_lo + (1 - alpha_lo) * ((i * 37) % 10) / 10
        ax.add_patch(Rectangle((x, y + i * h / n), w, h / n, fc=color, ec="white", lw=0.8, alpha=a, zorder=5))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=color, lw=1.2, zorder=6))


# ------------------------------------------------------------------ panel backgrounds
rbox(1, 2, 38, 78, "#f6f7f9", "#9aa5b1", lw=1.2, r=2)
rbox(42, 41, 61, 39, GREEN_BG, GREEN, lw=1.4, r=2)
rbox(42, 2, 61, 36, ORANGE_BG, ORANGE, lw=1.4, r=2)
rbox(106, 2, 43, 78, PURPLE_BG, PURPLE, lw=1.4, r=2)

txt(20, 77.3, "Inputs", 12, True)
txt(72.5, 77.3, "Spatial branch: two-layer graph convolution", 11.5, True, GREEN)
txt(72.5, 35.3, "Temporal branch: Transformer encoder", 11.5, True, ORANGE)
txt(127.5, 77.3, "Fusion and prediction", 12, True, PURPLE)

# ------------------------------------------------------------------ real well graph
c = pd.read_csv(CSV, parse_dates=["date"])
w = c.groupby("station_name").agg(lat=("latitude", "first"), lon=("longitude", "first"), d=("district_name", "first")).reset_index()
lat, lon = w.lat.values, w.lon.values
R = 6371.0
def hav(a, b, c_, d_):
    a, b, c_, d_ = map(np.radians, (a, b, c_, d_))
    x = np.sin((c_ - a) / 2) ** 2 + np.cos(a) * np.cos(c_) * np.sin((d_ - b) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(x))
D = np.array([[hav(lat[i], lon[i], lat[j], lon[j]) for j in range(len(w))] for i in range(len(w))])
gx0, gx1, gy0, gy1 = 4, 36, 50, 71
kx = np.cos(np.radians(lat.mean()))
px = gx0 + (lon - lon.min()) / (lon.max() - lon.min()) * (gx1 - gx0)
py = gy0 + (lat - lat.min()) / (lat.max() - lat.min()) * (gy1 - gy0)
for i in range(len(w)):
    for j in range(i + 1, len(w)):
        if D[i, j] < 11:
            ax.plot([px[i], px[j]], [py[i], py[j]], color="#7b8794", lw=0.9, alpha=0.75, zorder=3)
for i in range(len(w)):
    ax.add_patch(Circle((px[i], py[i]), 0.85, fc=DIST[w.d[i]], ec="white", lw=0.6, zorder=4))
for k,(nm,xx0) in enumerate([("Thanjavur",4.5),("Thiruvarur",13.8),("Nagapattinam",23.8)]):
    ax.add_patch(Circle((xx0,74.2),0.85,fc=DIST[nm],ec="white",lw=0.6,zorder=4))
    txt(xx0+1.3,74.2,nm,7.4,ha="left")
txt(20, 47.6, "Well graph  G = (V, E),  53 real wells", 9.8, True)
txt(20, 44.6, r"Edge weight  $A_{ij}=\exp(-d_{ij}^{2}/\sigma^{2})$,   $\tilde{A}=D^{-1/2}AD^{-1/2}$", 8.6)
txt(20, 42.1, "(links drawn for wells closer than 11 km)", 7.8, color="#52606d", style="italic")

# ------------------------------------------------------------------ sequence tokens
txt(20, 37.3, r"Per-well sequence  $X_i\in\mathbb{R}^{L\times F}$", 9.8, True)
cols = ["#9cc3e6", "#7fb0dc", "#b7d3ec", "#6aa1d4", "#a7c9e8", "#8bb8e0"]
x0, cw, gap = 4.2, 3.0, 0.55
for t in range(8):
    xx = x0 + t * (cw + gap)
    for f in range(6):
        a = 0.45 + 0.55 * (((t + 2) * (f + 3) * 7) % 10) / 10
        ax.add_patch(Rectangle((xx, 24 + f * 1.9), cw, 1.9, fc=cols[f], ec="white", lw=0.8, alpha=a, zorder=5))
    ax.add_patch(Rectangle((xx, 24), cw, 11.4, fc="none", ec=BLUE, lw=1.0, zorder=6))
    lab = "t" if t == 7 else f"t−{7 - t}"
    txt(xx + cw / 2, 22.4, lab, 7.6)
txt(20, 19.2, "8 most recent quarterly readings", 8.8)
txt(20, 16.4, "6 features per reading:", 8.4, color="#52606d")
txt(20, 13.9, "rainfall, Tmax, Tmin, humidity, ET, previous depth", 7.9, color="#52606d")

# ------------------------------------------------------------------ GCN drawing
def layer(xc, ys, color, fill):
    for yy in ys:
        ax.add_patch(Circle((xc, yy), 1.35, fc=fill, ec=color, lw=1.3, zorder=5))
ys = [47.5, 52.5, 57.5, 62.5, 67.5]
xs_l = [49, 64, 79]
rng = np.random.RandomState(3)
for k in range(2):
    for i, y0 in enumerate(ys):
        for j, y1 in enumerate(ys):
            if abs(i - j) <= 1 or rng.rand() < 0.18:
                ax.plot([xs_l[k] + 1.35, xs_l[k + 1] - 1.35], [y0, y1], color=GREEN, lw=0.8, alpha=0.45, zorder=3)
fills = ["#ffffff", "#bfe3d0", "#7cc4a0"]
for k, xc in enumerate(xs_l):
    layer(xc, ys, GREEN, fills[k])
txt(49, 43.7, r"$H^{(0)}=X$", 9)
txt(64, 43.7, r"$H^{(1)}=\sigma(\tilde{A}H^{(0)}W^{(0)})$", 8.6)
txt(79, 43.7, r"$H^{(2)}$", 9)
txt(56.5, 71.0, "GCN layer 1", 9.2, True, GREEN)
txt(71.5, 71.0, "GCN layer 2", 9.2, True, GREEN)
bar(91, 50, 3.4, 17, GREEN, n=8)
txt(92.7, 47.3, r"$h_{\mathrm{spatial}}$", 10)
arr((81, 57.5), (90.6, 57.5), color=GREEN)

# ------------------------------------------------------------------ temporal: tokens + positional encoding
txt(59, 32.2, "Input tokens (one per reading)", 8.6, color="#52606d")
for t in range(8):
    xx = 45.5 + t * 3.6
    ax.add_patch(Rectangle((xx, 28.2), 3.0, 2.4, fc="#f6d9b5", ec=ORANGE, lw=1.0, zorder=5))
xs = np.linspace(80.5, 94, 60)
ax.plot(xs, 29.4 + 1.4 * np.sin((xs - 80.5) / 13.5 * 4 * np.pi), color=ORANGE, lw=1.4, zorder=5)
arr((83.0, 29.4), (78.5, 29.4), color=ORANGE, lw=1.2) if False else None
ax.add_patch(Circle((77.0, 29.4), 1.4, fc="white", ec=ORANGE, lw=1.3, zorder=6))
txt(77.0, 29.4, "+", 10, True, ORANGE)
arr((73.9, 29.4), (75.5, 29.4), color=ORANGE, lw=1.2)
txt(86, 32.2, "Sinusoidal positional encoding", 8.2, color="#52606d")
# encoder block
rbox(44, 5, 45, 20, "white", ORANGE, lw=1.6, r=1.4, z=2)
txt(83.5, 26.6, "×2 layers", 9, True, ORANGE)
arr((77.0, 28.0), (77.0, 25.2), color=ORANGE, lw=1.4)
# attention heatmap (illustrative)
rs = np.random.RandomState(8)
M = rs.rand(8, 8) ** 2
M = M + np.eye(8) * 0.5 + np.roll(np.eye(8), 4, 1) * 0.3
M = np.tril(np.ones((8, 8)))[:, ::-1] * 0 + M
hx, hy, hs = 47.0, 8.6, 1.3
cm = plt.get_cmap("Oranges")
for i in range(8):
    for j in range(8):
        ax.add_patch(Rectangle((hx + j * hs, hy + (7 - i) * hs), hs, hs, fc=cm(0.1 + 0.8 * M[i, j] / M.max()), ec="white", lw=0.4, zorder=5))
txt(hx + 4 * hs, hy + 8 * hs + 3.4, "Self-attention", 8.6, True)
txt(hx + 4 * hs, hy + 8 * hs + 1.4, "(4 heads)", 8.0)
txt(hx + 4 * hs, hy - 1.7, "illustrative weights", 7.2, color="#52606d", style="italic")
for k, (lab, yy) in enumerate([("Add & Norm", 20.8), ("Feed-forward", 16.3), ("Add & Norm", 11.8)]):
    rbox(66, yy - 1.7, 20, 3.8, "#fdf2e4", ORANGE, lw=1.1, r=0.8, z=3)
    txt(76, yy + 0.2, lab, 8.6)
txt(76, 8.2, "dropout 0.2", 8.0, color="#52606d")
bar(93.2, 8, 3.4, 17, ORANGE, n=8)
txt(94.9, 5.3, r"$h_{\mathrm{temporal}}$", 10)
arr((89, 16.5), (92.8, 16.5), color=ORANGE)

# input -> branches
arr((36.5, 60), (42.2, 60), color=INK, lw=1.7)
txt(39.3, 62.0, r"$\tilde{A}$", 9.5)
arr((36.5, 29.5), (45.0, 29.4), color=INK, lw=1.7)
line([(36.5, 33), (40.2, 33), (40.2, 48.5), (45.4, 48.5)], color=BLUE, lw=1.4)
txt(38.9, 40.5, r"$X$", 9.5, color=BLUE, style="italic")

# ------------------------------------------------------------------ fusion / head / output
# concat
rbox(109, 50.5, 37, 22.5, "white", PURPLE, lw=1.4, r=1.2, z=2)
txt(127.5, 70.7, "Late fusion", 10.5, True, PURPLE)
bar(113, 55, 3.4, 12, GREEN, n=6)
bar(118, 55, 3.4, 12, ORANGE, n=6)
txt(116, 53.2, r"$h_s$", 8.6)
txt(121, 53.2, r"$h_t$", 8.6)
txt(131, 63.6, r"$u=W_f[\,h_s\,;\,h_t\,]+b_f$", 9.2)
txt(131, 59.4, "branches stay separate", 8.0, color="#52606d", style="italic")
txt(131, 57.0, "until this step", 8.0, color="#52606d", style="italic")
line([(97, 58), (101, 58), (101, 64), (112.6, 64)], color=GREEN, lw=1.6)
line([(97.5, 16.5), (104.5, 16.5), (104.5, 60), (112.6, 60)], color=ORANGE, lw=1.6)
# MLP head
txt(127.5, 46.6, "Prediction head", 10.5, True, PURPLE)
mlp_x = [116, 127, 138]
mlp_n = [4, 3, 1]
mlp_y = {4: [36, 39.5, 43, 32.5], 3: [34, 38, 42], 1: [38]}
mlp_y[4] = [33, 36.5, 40, 43.5]
for a, xa in enumerate(mlp_x[:-1]):
    for ya in mlp_y[mlp_n[a]]:
        for yb in mlp_y[mlp_n[a + 1]]:
            ax.plot([xa, mlp_x[a + 1]], [ya, yb], color=PURPLE, lw=0.7, alpha=0.5, zorder=3)
for xa, n in zip(mlp_x, mlp_n):
    for ya in mlp_y[n]:
        ax.add_patch(Circle((xa, ya), 1.3, fc="#d9c9ee" if xa < 138 else "#b79ad9", ec=PURPLE, lw=1.2, zorder=5))
arr((127.5, 50.5), (127.5, 48.4), color=PURPLE)
txt(127.5, 29.3, r"$\hat{y}_{i,T+1}=W_p\,u_{i,T}+b_p$", 9.4)
txt(127.5, 26.6, r"Loss: MSE $+\ \lambda\,\|\Theta\|_2^{2}$", 9.0)
arr((138, 36.4), (138, 31.2), color=PURPLE)

# output: real series of one well, last 8 quarters + target quarter
s = c.copy()
s["q"] = s.date.dt.to_period("Q")
s = s[s.station_name == "Lower Anaicut"].sort_values("date").groupby("q").last().reset_index().sort_values("q")
s = s[(s.q >= pd.Period("2022Q4")) & (s.q <= pd.Period("2024Q4"))]
vals = s.currentlevel.values
assert len(vals) == 9, len(vals)
ox0, ox1, oy0, oy1 = 113, 144, 7.0, 19.0
xx = np.linspace(ox0 + 1, ox1 - 5, 9)
yy = oy0 + 1 + (vals.max() - vals) / (vals.max() - vals.min()) * (oy1 - oy0 - 3)  # deeper water drawn lower
ax.plot([ox0, ox0, ox1], [oy1, oy0, oy0], color=INK, lw=1.0, zorder=4)
ax.plot(xx[:8], yy[:8], color=BLUE, lw=1.6, marker="o", ms=3.6, zorder=6)
ax.plot(xx[7:9], yy[7:9], color=PURPLE, lw=1.2, ls="--", zorder=5)
ax.plot(xx[8], yy[8], marker="*", ms=11, mfc="white", mec=PURPLE, mew=1.4, zorder=7, ls="none")
txt(xx[8] - 0.5, yy[8] + 2.8, r"target $y_{T+1}$", 8.2, color=PURPLE)
txt(128, 22.2, "Output: next-quarter depth (m)", 9.8, True)
txt(128, 4.9, "Example: Lower Anaicut, 8 observed quarters + target.", 7.4, color="#52606d", style="italic")
txt(128, 3.3, "Model output is compared with the target.", 7.4, color="#52606d", style="italic")
txt(ox0 - 1.3, (oy0 + oy1) / 2, "depth", 7.6, color="#52606d")
ax.texts[-1].set_rotation(90)

fig.savefig("/home/claude/fig3_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("/home/claude/fig3_architecture.pdf", bbox_inches="tight", facecolor="white")
print("ok")
