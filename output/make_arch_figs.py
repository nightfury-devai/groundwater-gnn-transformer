import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"

INK = "#1f2933"
C = {
    "in":  ("#eef3f8", "#4a6785"),   # inputs, blue-grey
    "sp":  ("#e6f4ee", "#2f7d5b"),   # spatial, green
    "tp":  ("#fdf1e3", "#b86a1d"),   # temporal, orange
    "fu":  ("#efe9f7", "#6b4c9a"),   # fusion, purple
    "ou":  ("#f3f3f3", "#555555"),
}


def box(ax, x, y, w, h, key, title=None, lines=(), tsize=11, bsize=9.2, lw=1.4, round_=1.2, title_dy=None, st=3.0):
    fc, ec = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    cur = y + h - (title_dy if title_dy else 2.6)
    if title is not None:
        ax.text(x + w / 2, cur, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=INK, zorder=3)
        cur -= 3.6
    elif lines:
        # centre the lines vertically
        total = (len(lines) - 1) * st
        cur = y + h / 2 + total / 2
    for t in lines:
        ax.text(x + w / 2, cur, t, ha="center", va="center", fontsize=bsize, color=INK, zorder=3)
        cur -= st


def arrow(ax, p0, p1, color=INK, lw=1.5, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=0, shrinkB=0))


def poly(ax, pts, color=INK, lw=1.5, arrow_end=True):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=4, solid_capstyle="butt")
    if arrow_end:
        arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


# ---------------------------------------------------------------- Figure 3
fig, ax = plt.subplots(figsize=(13, 6.6))
ax.set_xlim(0, 122)
ax.set_ylim(0, 62)
ax.axis("off")

box(ax, 2, 36, 27, 21, "in", "Well graph  G = (V, E)", [
    "Nodes: 53 wells",
    r"Edges: $A_{ij}=\exp(-d_{ij}^{2}/\sigma^{2})$",
    r"$d_{ij}$: haversine distance",
    r"$\tilde{A}=D^{-1/2}\,A\,D^{-1/2}$",
], st=3.4)
box(ax, 2, 4, 27, 27, "in", r"Per-well input  $X_i\in\mathbb{R}^{L\times F}$", [
    "Window: 8 most recent readings",
    "Rainfall (180-day mean)",
    "Tmax, Tmin (90-day mean)",
    "Humidity (90-day mean)",
    "Evapotranspiration (90-day sum)",
    "Previous depth to water level",
], st=3.2, bsize=9.0)

ec = C["sp"][1]
ax.add_patch(FancyBboxPatch((36, 35), 37, 22, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=ec, lw=1.8, zorder=1))
ax.text(54.5, 54.6, "Spatial branch (GCN)", ha="center", va="center", fontsize=11, fontweight="bold", color=ec, zorder=3)
box(ax, 38.5, 43.5, 32, 8, "sp", None, [r"GCN layer 1:  $H^{(1)}=\sigma(\tilde{A}\,X\,W^{(0)})$"], bsize=9.6)
box(ax, 38.5, 37, 32, 5, "sp", None, [r"GCN layer 2   $\rightarrow$   $h_{\mathrm{spatial}}$"], bsize=9.6)
arrow(ax, (54.5, 43.5), (54.5, 42), color=ec, lw=1.4)

ec2 = C["tp"][1]
ax.add_patch(FancyBboxPatch((36, 2), 37, 29, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=ec2, lw=1.8, zorder=1))
ax.text(54.5, 28.6, "Temporal branch (Transformer)", ha="center", va="center", fontsize=11, fontweight="bold", color=ec2, zorder=3)
box(ax, 38.5, 19.5, 32, 6.5, "tp", None, ["Sinusoidal positional encoding", r"+ linear embedding  ($d_h=64$)"], bsize=9.4, st=2.8)
box(ax, 38.5, 3.6, 32, 13.9, "tp", "Transformer encoder  ×2", [
    "Multi-head self-attention (4 heads)",
    "Add & Norm",
    "Feed-forward",
    r"Add & Norm   (dropout 0.2)",
], tsize=10, bsize=9.2, title_dy=2.2, st=2.1)
arrow(ax, (54.5, 19.5), (54.5, 17.5), color=ec2, lw=1.4)

# inputs -> branches
arrow(ax, (29, 49), (38.5, 49), color=INK)
ax.text(33.7, 50.2, r"$\tilde{A}$", fontsize=9.5, color=INK, ha="center", va="bottom")
arrow(ax, (29, 22), (38.5, 22.7), color=INK)
poly(ax, [(29, 28), (33.2, 28), (33.2, 45.6), (38.5, 45.6)], color="#4a6785", lw=1.4)
ax.text(34.0, 36.2, r"$X$", fontsize=9.5, color="#4a6785", ha="left", va="center", style="italic")

box(ax, 83, 41, 37, 16, "fu", "Late fusion", [
    r"$u_{i,t}=W_f\,[\,h_{\mathrm{spatial}}\,;\,h_{\mathrm{temporal}}\,]+b_f$",
    "Branches stay separate until this step",
], tsize=11, bsize=9.2, st=3.6)
box(ax, 83, 22, 37, 14, "fu", "Prediction head", [
    r"$\hat{y}_{i,T+1}=W_p\,u_{i,T}+b_p$",
    r"Loss:  MSE $+\ \lambda\,\|\Theta\|_2^{2}$",
], tsize=11, bsize=9.4, st=3.4)
box(ax, 83, 5, 37, 12, "ou", "Output", [
    "Depth to water level, next quarter (m)",
    "Per well, then district level",
], tsize=11, bsize=9.2, st=3.2)

poly(ax, [(73, 39.5), (77, 39.5), (77, 51.5), (83, 51.5)], color="#2f7d5b", lw=1.6)
ax.text(75, 38.2, r"$h_{\mathrm{spatial}}$", fontsize=9, color="#2f7d5b", ha="center", va="top")
poly(ax, [(73, 11), (80, 11), (80, 46), (83, 46)], color="#b86a1d", lw=1.6)
ax.text(76.5, 9.8, r"$h_{\mathrm{temporal}}$", fontsize=9, color="#b86a1d", ha="center", va="top")
arrow(ax, (101.5, 41), (101.5, 36), color=INK)
arrow(ax, (101.5, 22), (101.5, 17), color=INK)

fig.savefig("/home/claude/fig3_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("/home/claude/fig3_architecture.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
fig, ax = plt.subplots(figsize=(14, 3.9))
ax.set_xlim(0, 143)
ax.set_ylim(0, 34)
ax.axis("off")
stages = [
    ("in", "1  Data sources", ["CGWB water levels", "IMD rain, temperature", "NASA POWER: RH, ET"]),
    ("in", "2  Integration", ["Nearest-cell matching", "Trailing-window weather", "1,232 rows, 53 wells"]),
    ("sp", "3  Preparation", ["One reading per quarter", "Normalised windows", "70/15/15 time split"]),
    ("fu", "4  Model", ["Graph convolution branch", "Transformer branch", "Late fusion, linear head"]),
    ("tp", "5  Experiments", ["Baselines", "Ablations", "Robustness tests"]),
    ("ou", "6  Evaluation", ["MAE, RMSE, R²", "Per well, per district", "CGWB-style risk classes"]),
]
w, gap, x0 = 21.0, 2.6, 1.5
for i, (k, t, ls) in enumerate(stages):
    x = x0 + i * (w + gap)
    box(ax, x, 11, w, 19, k, t, ls, tsize=10.5, bsize=8.6, title_dy=3.0, st=3.8)
    if i < len(stages) - 1:
        arrow(ax, (x + w, 19.5), (x + w + gap, 19.5), color=INK, lw=1.5)
tags = ["Study Area and Data", "Study Area and Data", "Methodology", "Methodology", "Experimental Setup", "Results"]
for i, t in enumerate(tags):
    x = x0 + i * (w + gap)
    ax.text(x + w / 2, 7.0, t, ha="center", va="center", fontsize=9, color="#555555", style="italic")
fig.savefig("/home/claude/fig4_workflow.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("/home/claude/fig4_workflow.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("ok")
