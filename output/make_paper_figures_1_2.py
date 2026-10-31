"""
Paper Figure 1 (study area) and Figure 2 (data integration), regenerated from
the extended 2013-2025 data. Reads ONLY:

    data/cgwb_thanjavur_thiruvarur_nagapattinam_2013_2025.csv   (wells, district)
    data/imd_rain_thanjavur_thiruvarur_nagapattinam_2013_2025.csv  (rain cell centres)
    data/imd_temp_thanjavur_thiruvarur_nagapattinam_2013_2025.csv  (temp cell centres)
    archive/data_build/tn_district_boundaries.gpkg                (district outlines)

It does not import merge.py and does not recompute any weather feature. Matching
is the haversine nearest cell (R = 6371 km), computed per well and independently
for rain and temperature -- the same method as the merge.

Writes (existing figures are never touched):
    figures/fig1_study_area.png / .pdf
    figures/fig2_data_integration.png / .pdf

Run with the project .venv:  python -B -I output/make_paper_figures_1_2.py
"""

import math
import os

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")
CGWB = f"{DATA}/cgwb_thanjavur_thiruvarur_nagapattinam_2013_2025.csv"
RAIN = f"{DATA}/imd_rain_thanjavur_thiruvarur_nagapattinam_2013_2025.csv"
TEMP = f"{DATA}/imd_temp_thanjavur_thiruvarur_nagapattinam_2013_2025.csv"
BOUNDARIES = f"{ROOT}/archive/data_build/tn_district_boundaries.gpkg"

DPI = 300
R_EARTH_KM = 6371.0
# One map extent for both figures. It is wider than the district outlines so the
# centres of all used 1-degree temperature cells (10.5N 78.5E and 11.5N 79.5E)
# are on the map, and there is open sea on the east side for the north arrow.
XLIM = (78.40, 80.05)
YLIM = (10.05, 11.55)
LAT0 = 10.8
ASPECT = 1.0 / math.cos(math.radians(LAT0))

# Okabe-Ito (colour-blind safe) + distinct marker shapes
DISTRICTS = ["Thanjavur", "Thiruvarur", "Nagapattinam"]
COLOR = {"Thanjavur": "#E69F00", "Thiruvarur": "#0072B2", "Nagapattinam": "#009E73"}
LABEL_COLOR = {"Thanjavur": "#8A5A00", "Thiruvarur": "#004C7A", "Nagapattinam": "#00664A"}
MARKER = {"Thanjavur": "o", "Thiruvarur": "s", "Nagapattinam": "^"}

NOTE = ("District outlines as in the boundary file; Nagapattinam includes\n"
        "the Mayiladuthurai area (separated in 2020)")


# ----------------------------------------------------------------- data
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def load():
    wells = (pd.read_csv(CGWB).drop_duplicates("station_name")
             [["station_name", "district_name", "latitude", "longitude"]]
             .rename(columns={"station_name": "well", "district_name": "district",
                              "latitude": "lat", "longitude": "lon"})
             .sort_values(["district", "well"]).reset_index(drop=True))
    assert len(wells) == 53
    bnd = gpd.read_file(BOUNDARIES)
    cells = {}
    for key, path, prefix in (("rain", RAIN, "G"), ("temp", TEMP, "T")):
        c = (pd.read_csv(path, usecols=["grid_lat", "grid_long"]).drop_duplicates()
             .rename(columns={"grid_lat": "lat", "grid_long": "lon"})
             .sort_values(["lat", "lon"], ascending=[False, True]).reset_index(drop=True))
        c["id"] = [f"{prefix}{i + 1}" for i in range(len(c))]   # same G-numbering as the earlier maps
        cells[key] = c
    return wells, bnd, cells


def match(wells, cells):
    out = {}
    for key, c in cells.items():
        d = np.array([haversine_km(w.lat, w.lon, c["lat"].to_numpy(), c["lon"].to_numpy()) for w in wells.itertuples()])
        j = d.argmin(axis=1)
        out[key] = {"idx": j, "dist": d[np.arange(len(wells)), j], "used": sorted(set(j.tolist()))}
    return out


# ----------------------------------------------------------------- drawing helpers
def new_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect(ASPECT, adjustable="box")
    ax.set_xticks([78.5, 79.0, 79.5, 80.0])
    ax.set_xticklabels([f"{v:.1f}\N{DEGREE SIGN}E" for v in (78.5, 79.0, 79.5, 80.0)], fontsize=7)
    ax.set_yticks([10.5, 11.0, 11.5])
    ax.set_yticklabels([f"{v:.1f}\N{DEGREE SIGN}N" for v in (10.5, 11.0, 11.5)], fontsize=7)
    ax.set_xticks([78.75, 79.25, 79.75], minor=True)
    ax.set_yticks([10.25, 10.75, 11.25], minor=True)
    ax.tick_params(which="both", direction="out", length=3, width=0.6)
    ax.tick_params(which="minor", length=2)
    ax.grid(True, which="major", color="#000000", alpha=0.08, linewidth=0.5)
    ax.set_facecolor("#F6F7F9")
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    return ax


def draw_outlines(ax, bnd, per_district_color=True, fill_alpha=0.22):
    # light RGBA fill + thin dark edge in one pass, so the edge stays crisp
    for r in bnd.itertuples():
        face = to_rgba(COLOR[r.district_name] if per_district_color else "#9AA0A6", fill_alpha)
        gpd.GeoSeries([r.geometry], crs=bnd.crs).plot(
            ax=ax, facecolor=[face], edgecolor="#2B2B2B", linewidth=0.7, zorder=1)


def scale_bar(ax, km=25, x0=79.58, y=10.12):
    dlon = km / (111.32 * math.cos(math.radians(LAT0)))
    ax.add_patch(Rectangle((x0, y), dlon / 2, 0.012, facecolor="black", edgecolor="black", linewidth=0.5, zorder=6))
    ax.add_patch(Rectangle((x0 + dlon / 2, y), dlon / 2, 0.012, facecolor="white", edgecolor="black", linewidth=0.5, zorder=6))
    for x, t in ((x0, "0"), (x0 + dlon, f"{km} km")):
        ax.text(x, y + 0.03, t, ha="center", va="bottom", fontsize=6.5, zorder=6)


def north_arrow(ax, x=79.97, y0=11.33, y1=11.46):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1, mutation_scale=9))
    ax.text(x, y1 + 0.012, "N", ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=6)


def marker_px(ax, size_pt2):
    return math.sqrt(size_pt2) * ax.figure.dpi / 72.0 / 2.0


def overlaps(b, others):
    return [o for o in others if b.overlaps(o)]


# ----------------------------------------------------------------- Figure 1
def figure1(wells, bnd, m):
    fig = plt.figure(figsize=(7, 6.5), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = new_axes(fig, [0.075, 0.075, 0.915, 0.90])
    draw_outlines(ax, bnd, per_district_color=True, fill_alpha=0.22)

    S = 30
    handles = []
    for d in DISTRICTS:
        w = wells[wells.district == d]
        ax.scatter(w.lon, w.lat, s=S, marker=MARKER[d], facecolor=COLOR[d], edgecolor="#1A1A1A",
                   linewidth=0.5, zorder=4)
        handles.append(Line2D([0], [0], marker=MARKER[d], linestyle="", markersize=6.5, markerfacecolor=COLOR[d],
                              markeredgecolor="#1A1A1A", markeredgewidth=0.5, label=f"{d} (n = {len(w)})"))
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.988), frameon=True, fontsize=7.5,
                    title="CGWB wells by district", title_fontsize=7.5, borderpad=0.7, labelspacing=0.5,
                    framealpha=0.95, edgecolor="#9A9A9A")
    leg.set_zorder(7)
    note = ax.text(0.012, 0.012, NOTE, transform=ax.transAxes, fontsize=6.2, va="bottom", ha="left",
                   style="italic", color="#333333", zorder=7)
    scale_bar(ax)
    north_arrow(ax)

    # district names: once each, away from wells, legend, note
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    wpx = ax.transData.transform(wells[["lon", "lat"]].to_numpy())
    mr = marker_px(ax, S) + 2.5
    obstacles = [Bbox.from_extents(x - mr, y - mr, x + mr, y + mr) for x, y in wpx]
    fixed = [leg.get_window_extent(rnd).padded(4), note.get_window_extent(rnd).padded(4)]
    placed, report = [], {}
    geoms = {r.district_name: r.geometry for r in bnd.itertuples()}
    for d in ["Thanjavur", "Nagapattinam", "Thiruvarur"]:
        g = geoms[d]
        minx, miny, maxx, maxy = g.bounds
        xs, ys = np.meshgrid(np.arange(minx, maxx, 0.01), np.arange(miny, maxy, 0.01))
        xs, ys = xs.ravel(), ys.ravel()
        inside = shapely.contains_xy(g, xs, ys)
        in_other = np.zeros(len(xs), bool)
        for o, og in geoms.items():
            if o != d:
                in_other |= shapely.contains_xy(og, xs, ys)
        pts = shapely.points(xs, ys)
        d_bnd_km = shapely.distance(g.boundary, pts) * 111.0
        d_well_km = np.min(haversine_km(ys[:, None], xs[:, None], wells.lat.to_numpy()[None, :], wells.lon.to_numpy()[None, :]), axis=1)
        score = np.minimum(d_well_km, 10) + 0.4 * np.minimum(d_bnd_km, 8) - np.where(inside, 0, 12) - np.where(in_other, 25, 0)
        # candidates: horizontal and vertical text. A candidate is valid if its box touches no
        # well, legend, note or placed label; among valid ones keep the one whose box lies
        # most fully inside its own district (and least inside another district).
        order = np.argsort(-score)[:260]
        inv = ax.transData.inverted()
        best = None
        for rot in (0, 90):
            for k in order:
                t = ax.text(xs[k], ys[k], d, ha="center", va="center", fontsize=9.5, fontweight="bold",
                            rotation=rot, rotation_mode="anchor", color=LABEL_COLOR[d], zorder=5,
                            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
                b = t.get_window_extent(rnd).padded(3)
                ok = not overlaps(b, obstacles + fixed + placed)
                frac_own = frac_other = 0.0
                if ok:
                    gx, gy = np.meshgrid(np.linspace(b.x0, b.x1, 14), np.linspace(b.y0, b.y1, 5))
                    dpts = inv.transform(np.column_stack([gx.ravel(), gy.ravel()]))
                    frac_own = float(shapely.contains_xy(g, dpts[:, 0], dpts[:, 1]).mean())
                    oth = np.zeros(len(dpts), bool)
                    for o, og in geoms.items():
                        if o != d:
                            oth |= shapely.contains_xy(og, dpts[:, 0], dpts[:, 1])
                    frac_other = float(oth.mean())
                t.remove()
                if not ok:
                    continue
                val = frac_own - 0.6 * frac_other + (0.03 if rot == 0 else 0.0)
                if best is None or val > best["val"]:
                    best = {"val": val, "x": float(xs[k]), "y": float(ys[k]), "rot": rot, "b": b,
                            "frac_own": frac_own, "frac_other": frac_other}
        if best is None:
            raise SystemExit(f"could not place the {d} label without overlap")
        ax.text(best["x"], best["y"], d, ha="center", va="center", fontsize=9.5, fontweight="bold",
                rotation=best["rot"], rotation_mode="anchor", color=LABEL_COLOR[d], zorder=5,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        placed.append(best["b"])
        report[d] = {"pos": (round(best["x"], 3), round(best["y"], 3)), "rotation": best["rot"],
                     "share_of_text_box_inside_own_district": round(best["frac_own"], 2),
                     "share_inside_other_district": round(best["frac_other"], 2), "overlaps": 0}

    fig.savefig(f"{FIG_DIR}/fig1_study_area.png", dpi=DPI, facecolor="white")
    fig.savefig(f"{FIG_DIR}/fig1_study_area.pdf", facecolor="white")
    plt.close(fig)
    return report


# ----------------------------------------------------------------- Figure 2
def panel(ax, fig, wells, bnd, cells, mt, half, title_lines, footnote=None):
    draw_outlines(ax, bnd, per_district_color=False, fill_alpha=0.30)
    used = set(mt["used"])
    c = cells
    # cell footprints (centre +/- half the grid step)
    for i, r in c.iterrows():
        x0, y0 = r.lon - half, r.lat - half
        if i in used:
            ax.add_patch(Rectangle((x0, y0), 2 * half, 2 * half, facecolor="#56B4E9", alpha=0.32, edgecolor="none", zorder=2))
            ax.add_patch(Rectangle((x0, y0), 2 * half, 2 * half, facecolor="none", edgecolor="#0072B2", linewidth=1.0, zorder=3))
        else:
            ax.add_patch(Rectangle((x0, y0), 2 * half, 2 * half, facecolor="none", edgecolor="#7A7A7A",
                                   linewidth=0.8, linestyle=(0, (3, 2)), zorder=3))
    # match lines well -> matched cell centre
    segs = [[(w.lon, w.lat), (c.lon[mt["idx"][k]], c.lat[mt["idx"][k]])] for k, w in enumerate(wells.itertuples())]
    ax.add_collection(LineCollection(segs, colors="#4D4D4D", linewidths=0.55, alpha=0.75, zorder=4))
    ax.scatter(c.lon[list(used)], c.lat[list(used)], s=26, marker="+", color="#00355A", linewidths=1.1, zorder=5)
    S = 9
    ax.scatter(wells.lon, wells.lat, s=S, color="#1A1A1A", zorder=6, linewidths=0)
    scale_bar(ax, x0=79.55)

    # cell labels at a free corner of the visible part of each cell
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    wpx = ax.transData.transform(wells[["lon", "lat"]].to_numpy())
    mr = marker_px(ax, S) + 1.5
    obstacles = [Bbox.from_extents(x - mr, y - mr, x + mr, y + mr) for x, y in wpx]
    placed, unresolved = [], []
    for i, r in c.iterrows():
        x0, x1 = max(r.lon - half, XLIM[0]), min(r.lon + half, XLIM[1])
        y0, y1 = max(r.lat - half, YLIM[0]), min(r.lat + half, YLIM[1])
        if x1 - x0 < 0.02 or y1 - y0 < 0.02:
            continue                                     # cell not visible on this map
        inset = 0.012
        corners = [(x0 + inset, y1 - inset, "left", "top"), (x1 - inset, y1 - inset, "right", "top"),
                   (x0 + inset, y0 + inset, "left", "bottom"), (x1 - inset, y0 + inset, "right", "bottom")]
        ok = False
        for (x, y, ha, va) in corners:
            t = ax.text(x, y, r.id, ha=ha, va=va, fontsize=6.3, color="#0B3C5D" if i in used else "#555555",
                        fontweight="bold" if i in used else "normal", zorder=7,
                        path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
            b = t.get_window_extent(rnd).padded(1.5)
            if not overlaps(b, obstacles + placed):
                placed.append(b)
                ok = True
                break
            t.remove()
        if not ok:
            # keep the first corner and report it
            x, y, ha, va = corners[0]
            t = ax.text(x, y, r.id, ha=ha, va=va, fontsize=6.3, zorder=7,
                        path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
            placed.append(t.get_window_extent(rnd).padded(1.5))
            unresolved.append(r.id)
    ax.set_title("\n".join(title_lines), fontsize=8.3, loc="left", pad=6)
    if footnote:
        ax.text(0.012, 0.012, footnote, transform=ax.transAxes, fontsize=6.0, va="bottom", ha="left",
                color="#333333", style="italic", zorder=8,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
    return unresolved


def figure2(wells, bnd, cells, m):
    fig = plt.figure(figsize=(12, 6), dpi=DPI)
    fig.patch.set_facecolor("white")
    axa = new_axes(fig, [0.045, 0.14, 0.44, 0.74])
    axb = new_axes(fig, [0.525, 0.14, 0.44, 0.74])

    r, t = m["rain"], m["temp"]
    ta = [f"(a) Rainfall: IMD 0.25\N{DEGREE SIGN} grid, {len(r['used'])} of {len(cells['rain'])} cells used",
          f"median well\N{EN DASH}cell distance {np.median(r['dist']):.1f} km"]
    unused_t = [cells["temp"].id[i] for i in range(len(cells["temp"])) if i not in set(t["used"])]
    off = [i for i in unused_t
           if (lambda row: (row.lon + 0.5 < XLIM[0] or row.lon - 0.5 > XLIM[1] or row.lat + 0.5 < YLIM[0] or row.lat - 0.5 > YLIM[1]))(
               cells["temp"][cells["temp"].id == i].iloc[0])]
    tb = [f"(b) Temperature: IMD 1\N{DEGREE SIGN} grid, {len(t['used'])} of {len(cells['temp'])} cells used",
          f"median well\N{EN DASH}cell distance {np.median(t['dist']):.1f} km"]
    foot = ("Cells " + ", ".join(off) + " are outside the map extent.") if off else None
    un_a = panel(axa, fig, wells, bnd, cells["rain"], r, 0.125, ta)
    un_b = panel(axb, fig, wells, bnd, cells["temp"], t, 0.5, tb, footnote=foot)

    handles = [Patch(facecolor="#56B4E9", alpha=0.45, edgecolor="#0072B2", label="IMD cell used (nearest to at least one well)"),
               Patch(facecolor="none", edgecolor="#7A7A7A", linestyle=(0, (3, 2)), label="IMD cell not used"),
               Line2D([0], [0], marker="o", linestyle="", markersize=3.6, color="#1A1A1A", label="CGWB well"),
               Line2D([0], [0], color="#4D4D4D", linewidth=0.8, label="well to matched cell centre (+)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, 0.015))
    fig.savefig(f"{FIG_DIR}/fig2_data_integration.png", dpi=DPI, facecolor="white")
    fig.savefig(f"{FIG_DIR}/fig2_data_integration.pdf", facecolor="white")
    plt.close(fig)
    return {"rain_labels_unresolved": un_a, "temp_labels_unresolved": un_b, "temp_cells_outside_extent": off}


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    wells, bnd, cells = load()
    m = match(wells, cells)

    print("=== CHECK ===")
    print("wells per district:", wells.district.value_counts().to_dict(), "| total", len(wells))
    for k, name in (("rain", "rainfall (0.25 deg)"), ("temp", "temperature (1 deg)")):
        d = m[k]["dist"]
        used_ids = [cells[k].id[i] for i in m[k]["used"]]
        print(f"{name}: cells used {len(m[k]['used'])} of {len(cells[k])} -> {used_ids}")
        print(f"   match distance km: median {np.median(d):.2f} | mean {d.mean():.2f} | max {d.max():.2f} | min {d.min():.2f}")
    exp = {"rain": (14, 18, 11.1), "temp": (3, 9, 44.4)}
    for k, (u, n, med) in exp.items():
        got = (len(m[k]["used"]), len(cells[k]), round(float(np.median(m[k]["dist"])), 1))
        print(f"expected {k}: used {u}/{n}, median {med} km -> got {got[0]}/{got[1]}, {got[2]} km",
              "(matches)" if got == (u, n, med) else "(DIFFERS)")

    r1 = figure1(wells, bnd, m)
    print("Figure 1 district label positions (lon, lat):", r1)
    r2 = figure2(wells, bnd, cells, m)
    print("Figure 2 label check:", r2)
    print("written: fig1_study_area.png/.pdf, fig2_data_integration.png/.pdf in", FIG_DIR)


if __name__ == "__main__":
    main()
