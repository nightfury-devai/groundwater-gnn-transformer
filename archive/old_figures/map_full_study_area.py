"""
Full study area: all 53 wells + all 18 IMD rainfall grid cells across
Thanjavur, Thiruvarur, and Nagapattinam, with all three district
boundaries outlined and labeled.

Read-only visualization, new/additional output. Does NOT touch
merge.py, merged_final.csv, or any existing CSV output. Imports
merge.py's own haversine_km() / nearest_cell() so the matching logic
here is identical to what Phase B actually used, not a separately-
written copy. Same tile-fetching approach as map_thanjavur_wells.py
(contextily's rasterio dependency is blocked by this machine's Windows
Application Control policy -- see that script's docstring for detail),
so Esri World Imagery tiles are fetched and stitched directly here too.

Output:
    figures/full_study_area_map.png -- the map, 300+ DPI
"""

import math
import os
import sys
from io import BytesIO

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from pyproj import Transformer
from shapely.geometry import Point

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

sys.path.insert(0, DATA_DIR)
from merge import haversine_km, nearest_cell  # noqa: E402

CGWB_PATH = os.path.join(DATA_DIR, "cgwb_thanjavur_thiruvarur_nagapattinam.csv")
RAIN_PATH = os.path.join(DATA_DIR, "imd_rain_thanjavur_thiruvarur_nagapattinam.csv")
BOUNDARIES_PATH = os.path.join(DATA_DIR, "tn_district_boundaries.gpkg")
MAP_OUT = os.path.join(FIGURES_DIR, "full_study_area_map.png")

DISTRICT_COLORS = {
    "Thanjavur": "#f5e400",     # yellow
    "Nagapattinam": "#00e5ff",  # cyan
    "Thiruvarur": "#ff2fd6",    # magenta
}

ESRI_TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TILE_SIZE = 256
_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform


def deg2tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def fetch_esri_basemap(lat_min, lat_max, lon_min, lon_max, zoom):
    x0, y0 = deg2tile(lat_max, lon_min, zoom)
    x1, y1 = deg2tile(lat_min, lon_max, zoom)

    xmin_t, xmax_t = min(x0, x1), max(x0, x1)
    ymin_t, ymax_t = min(y0, y1), max(y0, y1)

    n_cols = xmax_t - xmin_t + 1
    n_rows = ymax_t - ymin_t + 1

    mosaic = Image.new("RGB", (n_cols * TILE_SIZE, n_rows * TILE_SIZE))

    for col, xt in enumerate(range(xmin_t, xmax_t + 1)):
        for row, yt in enumerate(range(ymin_t, ymax_t + 1)):
            url = ESRI_TILE_URL.format(z=zoom, y=yt, x=xt)
            resp = requests.get(url, timeout=20, headers={"User-Agent": "groundwater-project-thesis-map/1.0"})
            resp.raise_for_status()
            tile_img = Image.open(BytesIO(resp.content)).convert("RGB")
            mosaic.paste(tile_img, (col * TILE_SIZE, row * TILE_SIZE))

    lat_top, lon_left = tile2deg(xmin_t, ymin_t, zoom)
    lat_bottom, lon_right = tile2deg(xmax_t + 1, ymax_t + 1, zoom)

    x_left, y_top = _to_3857(lon_left, lat_top)
    x_right, y_bottom = _to_3857(lon_right, lat_bottom)

    return np.asarray(mosaic), (x_left, x_right, y_bottom, y_top)


# Candidate label offsets, in points, ordered by preference: right, then
# the seven other compass directions, at increasing radius if the closer
# ring is fully occupied. No scipy/adjustText dependency (both blocked on
# this machine, see below) -- plain greedy placement using matplotlib's
# own renderer for real text bounding boxes.
_ANGLES_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
_RADII_PT = [7, 11, 16, 22]
CANDIDATE_OFFSETS = [
    (round(r * math.cos(math.radians(a)), 1), round(r * math.sin(math.radians(a)), 1))
    for r in _RADII_PT for a in _ANGLES_DEG
]


def _inflate(bbox, pad=1.5):
    return bbox.expanded(1.0, 1.0).padded(pad)


def _overlaps_any(bbox, others):
    return any(bbox.overlaps(o) for o in others)


def place_well_labels(fig, ax, wells_gdf, cells_gdf, fontsize=5.2):
    """Greedy label placement: for each well, try candidate offsets in
    order of preference and keep the first one whose rendered text bbox
    doesn't overlap any marker or already-placed label. Falls back to the
    first candidate (closest, straight right) if every option collides,
    and reports that well as unresolved.

    Returns (repositioned, unresolved) -- lists of well names that needed
    a non-default offset, and that still overlapped after exhausting all
    candidates, respectively.
    """
    fig.canvas.draw()  # required once so renderer can compute real text extents
    renderer = fig.canvas.get_renderer()

    placed_bboxes = []

    # seed with marker footprints (wells + grid cells) and grid-cell label
    # text extents, already drawn onto the axes, so well labels avoid both
    for artist in ax.texts:
        placed_bboxes.append(_inflate(artist.get_window_extent(renderer=renderer)))

    marker_radius_pt = 7  # approx on-screen radius of the marker dots/squares
    for gdf in (wells_gdf, cells_gdf):
        for geom in gdf.geometry:
            cx, cy = ax.transData.transform((geom.x, geom.y))
            from matplotlib.transforms import Bbox
            placed_bboxes.append(Bbox.from_extents(
                cx - marker_radius_pt, cy - marker_radius_pt,
                cx + marker_radius_pt, cy + marker_radius_pt,
            ))

    repositioned, unresolved = [], []

    for row in wells_gdf.itertuples():
        placed = False
        for i, (dx, dy) in enumerate(CANDIDATE_OFFSETS):
            ann = ax.annotate(
                row.well_name, (row.geometry.x, row.geometry.y),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=fontsize, color="white",
                path_effects=[pe.withStroke(linewidth=1.6, foreground="black")],
            )
            bbox = ann.get_window_extent(renderer=renderer)
            if not _overlaps_any(bbox, placed_bboxes):
                placed_bboxes.append(_inflate(bbox))
                if i > 0:
                    repositioned.append(row.well_name)
                placed = True
                break
            ann.remove()

        if not placed:
            # every candidate collided -- fall back to the default position
            dx, dy = CANDIDATE_OFFSETS[0]
            ann = ax.annotate(
                row.well_name, (row.geometry.x, row.geometry.y),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=fontsize, color="white",
                path_effects=[pe.withStroke(linewidth=1.6, foreground="black")],
            )
            placed_bboxes.append(_inflate(ann.get_window_extent(renderer=renderer)))
            unresolved.append(row.well_name)

    return repositioned, unresolved


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    cgwb = pd.read_csv(CGWB_PATH)
    wells = cgwb.drop_duplicates(subset="station_name")[
        ["station_name", "district_name", "latitude", "longitude"]
    ].reset_index(drop=True)

    rain = pd.read_csv(RAIN_PATH)
    rain_cells = rain[["grid_lat", "grid_long"]].drop_duplicates().to_numpy()

    matches = []
    for well in wells.itertuples(index=False):
        cell_lat, cell_lon = nearest_cell(well.latitude, well.longitude, rain_cells)
        dist_km = haversine_km(well.latitude, well.longitude, cell_lat, cell_lon)
        matches.append((well.station_name, well.latitude, well.longitude, cell_lat, cell_lon, dist_km))

    match_df = pd.DataFrame(matches, columns=[
        "well_name", "well_lat", "well_lon", "grid_lat", "grid_lon", "distance_km",
    ])

    # ALL 18 IMD rain grid cells, not just matched ones (same approach as
    # the interactive map)
    all_cells = (
        rain[["grid_lat", "grid_long"]]
        .drop_duplicates()
        .rename(columns={"grid_long": "grid_lon"})
        .sort_values(["grid_lat", "grid_lon"], ascending=[False, True])
        .reset_index(drop=True)
    )
    all_cells["grid_id"] = ["G" + str(i + 1) for i in range(len(all_cells))]
    match_df = match_df.merge(all_cells, on=["grid_lat", "grid_lon"], how="left")

    boundaries = gpd.read_file(BOUNDARIES_PATH)

    wells_gdf = gpd.GeoDataFrame(
        match_df,
        geometry=[Point(xy) for xy in zip(match_df["well_lon"], match_df["well_lat"])],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    cells_gdf = gpd.GeoDataFrame(
        all_cells,
        geometry=[Point(xy) for xy in zip(all_cells["grid_lon"], all_cells["grid_lat"])],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    boundaries_3857 = boundaries.to_crs(epsg=3857)

    # bounding box: wells + grid cells + all three district boundaries
    b = boundaries.total_bounds  # (minx, miny, maxx, maxy) in EPSG:4326
    all_lats = pd.concat([match_df["well_lat"], all_cells["grid_lat"], pd.Series([b[1], b[3]])])
    all_lons = pd.concat([match_df["well_lon"], all_cells["grid_lon"], pd.Series([b[0], b[2]])])
    lat_min, lat_max = all_lats.min(), all_lats.max()
    lon_min, lon_max = all_lons.min(), all_lons.max()
    lat_buf = (lat_max - lat_min) * 0.08
    lon_buf = (lon_max - lon_min) * 0.08
    lat_min, lat_max = lat_min - lat_buf, lat_max + lat_buf
    lon_min, lon_max = lon_min - lon_buf, lon_max + lon_buf

    print(f"Fetching Esri World Imagery tiles for bbox ({lat_min:.3f}, {lon_min:.3f}) to ({lat_max:.3f}, {lon_max:.3f})...")
    basemap_img, extent_3857 = fetch_esri_basemap(lat_min, lat_max, lon_min, lon_max, zoom=10)

    xmin, ymin = _to_3857(lon_min, lat_min)
    xmax, ymax = _to_3857(lon_max, lat_max)

    fig, ax = plt.subplots(figsize=(12, 13))
    ax.imshow(basemap_img, extent=extent_3857, zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # district boundaries, distinctly colored + labeled
    for row in boundaries_3857.itertuples():
        color = DISTRICT_COLORS[row.district_name]
        gpd.GeoSeries([row.geometry], crs=boundaries_3857.crs).plot(
            ax=ax, facecolor="none", edgecolor=color, linewidth=2.2, zorder=1,
        )
        centroid = row.geometry.centroid
        ax.annotate(row.district_name, (centroid.x, centroid.y),
                    fontsize=13, fontweight="bold", color=color, ha="center",
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")], zorder=1)

    # connecting lines: well -> its matched grid cell
    for row in wells_gdf.itertuples():
        cell_row = cells_gdf[cells_gdf["grid_id"] == row.grid_id].iloc[0]
        ax.plot(
            [row.geometry.x, cell_row.geometry.x],
            [row.geometry.y, cell_row.geometry.y],
            color="white", linewidth=0.6, alpha=0.75, zorder=2,
        )

    wells_gdf.plot(ax=ax, color="#1f77ff", markersize=42, marker="o",
                    edgecolor="white", linewidth=0.7, zorder=3)
    cells_gdf.plot(ax=ax, color="#ff5a1f", markersize=110, marker="s",
                    edgecolor="white", linewidth=0.9, zorder=4)

    for row in cells_gdf.itertuples():
        ax.annotate(row.grid_id, (row.geometry.x, row.geometry.y),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, fontweight="bold", color="#ff5a1f",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    ax.set_axis_off()
    ax.set_title("Full Study Area: Thanjavur, Thiruvarur & Nagapattinam\nWell-to-Rainfall-Grid Spatial Matching",
                  fontsize=15, fontweight="bold", pad=14)

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77ff",
               markeredgecolor="white", markersize=9, label=f"CGWB well (n={len(wells_gdf)})"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#ff5a1f",
               markeredgecolor="white", markersize=10, label=f"IMD rainfall grid cell (n={len(cells_gdf)})"),
        Line2D([0], [0], color="white", linewidth=1.2, label="nearest-cell match"),
        Line2D([0], [0], color=DISTRICT_COLORS["Thanjavur"], linewidth=2.2, label="Thanjavur boundary"),
        Line2D([0], [0], color=DISTRICT_COLORS["Nagapattinam"], linewidth=2.2, label="Nagapattinam boundary"),
        Line2D([0], [0], color=DISTRICT_COLORS["Thiruvarur"], linewidth=2.2, label="Thiruvarur boundary"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", framealpha=0.85, fontsize=8.5)

    scalebar = ScaleBar(1, units="m", location="lower right", box_alpha=0.7, font_properties={"size": 9})
    ax.add_artist(scalebar)

    ax.annotate("N", xy=(0.96, 0.90), xytext=(0.96, 0.82), xycoords="axes fraction",
                fontsize=13, fontweight="bold", color="white", ha="center",
                arrowprops=dict(facecolor="white", edgecolor="white", width=3, headwidth=10, headlength=10),
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # layout must be finalized *before* well labels are placed, since label
    # placement below measures real pixel bounding boxes -- tight_layout
    # after that point would shift the axes and invalidate those positions
    fig.tight_layout()

    well_repositioned, well_unresolved = place_well_labels(fig, ax, wells_gdf, cells_gdf)

    fig.savefig(MAP_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved map: {MAP_OUT}")
    print(f"Total wells plotted: {len(wells_gdf)}")
    print(f"Total grid cells plotted: {len(cells_gdf)}")
    print(f"Districts: {', '.join(boundaries['district_name'].tolist())}")
    print(f"\nWell labels needing a non-default position to avoid collision: {len(well_repositioned)}")
    if well_repositioned:
        print("  " + ", ".join(well_repositioned))
    print(f"Well labels with unresolved overlap even after all candidates tried: {len(well_unresolved)}")
    if well_unresolved:
        print("  " + ", ".join(well_unresolved))


if __name__ == "__main__":
    main()
