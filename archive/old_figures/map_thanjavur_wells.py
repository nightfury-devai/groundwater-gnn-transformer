"""
Thanjavur district: well-to-rainfall-grid spatial matching map.

Read-only visualization. Does NOT touch merge.py or merged_final.csv --
imports merge.py's own haversine_km() / nearest_cell() so the matching
logic here is guaranteed identical to what Phase B actually used, not a
separately-written (and possibly drifting) copy.

Basemap note: contextily hard-imports rasterio at module load time, and
rasterio's compiled extension is blocked by this machine's Windows
Application Control policy (confirmed consistent, not transient -- same
class of issue as the earlier numpy/pandas block on the system Python,
which is why this project uses its own .venv). Rather than ship a map
without real satellite imagery, this script fetches and stitches Esri
World Imagery XYZ tiles directly over HTTP, bypassing contextily/rasterio
entirely. Visual result is the same: real satellite basemap underneath
the plotted points.

Output:
    figures/thanjavur_well_grid_map.png   -- the map, 300+ DPI
    figures/thanjavur_well_grid_match.csv -- well/grid/distance table
"""

import math
import os
import sys

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from pyproj import Transformer
from shapely.geometry import Point

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Import the exact same nearest-cell logic merge.py uses -- don't rewrite it.
sys.path.insert(0, DATA_DIR)
from merge import haversine_km, nearest_cell  # noqa: E402

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
    """Fetch and stitch Esri World Imagery tiles covering the given bbox.
    Returns (image_array, extent_3857) where extent_3857 is
    (xmin, xmax, ymin, ymax) in EPSG:3857 meters, ready for imshow."""
    x0, y0 = deg2tile(lat_max, lon_min, zoom)  # top-left
    x1, y1 = deg2tile(lat_min, lon_max, zoom)  # bottom-right

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

    # geographic bounds of the stitched mosaic
    lat_top, lon_left = tile2deg(xmin_t, ymin_t, zoom)
    lat_bottom, lon_right = tile2deg(xmax_t + 1, ymax_t + 1, zoom)

    x_left, y_top = _to_3857(lon_left, lat_top)
    x_right, y_bottom = _to_3857(lon_right, lat_bottom)

    extent_3857 = (x_left, x_right, y_bottom, y_top)
    return np.asarray(mosaic), extent_3857

CGWB_PATH = os.path.join(DATA_DIR, "cgwb_thanjavur_thiruvarur_nagapattinam.csv")
RAIN_PATH = os.path.join(DATA_DIR, "imd_rain_thanjavur_thiruvarur_nagapattinam.csv")
BOUNDARIES_PATH = os.path.join(DATA_DIR, "tn_district_boundaries.gpkg")

MAP_OUT = os.path.join(FIGURES_DIR, "thanjavur_well_grid_map.png")
CSV_OUT = os.path.join(FIGURES_DIR, "thanjavur_well_grid_match.csv")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    cgwb = pd.read_csv(CGWB_PATH)
    wells = cgwb.drop_duplicates(subset="station_name")
    wells = wells[wells["district_name"] == "Thanjavur"][
        ["station_name", "latitude", "longitude"]
    ].reset_index(drop=True)

    rain = pd.read_csv(RAIN_PATH)
    rain_cells = rain[["grid_lat", "grid_long"]].drop_duplicates().to_numpy()

    # Step 2: nearest rain cell + distance per well (same method as merge.py)
    matches = []
    for well in wells.itertuples(index=False):
        cell_lat, cell_lon = nearest_cell(well.latitude, well.longitude, rain_cells)
        dist_km = haversine_km(well.latitude, well.longitude, cell_lat, cell_lon)
        matches.append((well.station_name, well.latitude, well.longitude, cell_lat, cell_lon, dist_km))

    match_df = pd.DataFrame(matches, columns=[
        "well_name", "well_lat", "well_lon", "grid_lat", "grid_lon", "distance_km",
    ])

    # Step 3: distinct grid cells actually used, assigned stable IDs (sorted
    # north-to-south, then west-to-east, so G1 is always the same cell on reruns)
    distinct_cells = (
        match_df[["grid_lat", "grid_lon"]]
        .drop_duplicates()
        .sort_values(["grid_lat", "grid_lon"], ascending=[False, True])
        .reset_index(drop=True)
    )
    distinct_cells["grid_id"] = ["G" + str(i + 1) for i in range(len(distinct_cells))]

    match_df = match_df.merge(distinct_cells, on=["grid_lat", "grid_lon"], how="left")
    match_df = match_df[[
        "well_name", "well_lat", "well_lon", "grid_id", "grid_lat", "grid_lon", "distance_km",
    ]].sort_values("well_name").reset_index(drop=True)

    match_df.to_csv(CSV_OUT, index=False)

    # district boundary outline (no fill), reprojected to match the rest of the map
    boundaries = gpd.read_file(BOUNDARIES_PATH)
    thanjavur_boundary = boundaries[boundaries["district_name"] == "Thanjavur"].to_crs(epsg=3857)

    # --- Build the map ---
    wells_gdf = gpd.GeoDataFrame(
        match_df,
        geometry=[Point(xy) for xy in zip(match_df["well_lon"], match_df["well_lat"])],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    cells_gdf = gpd.GeoDataFrame(
        distinct_cells,
        geometry=[Point(xy) for xy in zip(distinct_cells["grid_lon"], distinct_cells["grid_lat"])],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    # bounding box (wells + grid cells + district boundary), buffered so
    # nothing is flush against the map edge or clipped
    boundary_bounds = boundaries[boundaries["district_name"] == "Thanjavur"].total_bounds  # (minx, miny, maxx, maxy) in EPSG:4326
    all_lats = pd.concat([
        match_df["well_lat"], distinct_cells["grid_lat"],
        pd.Series([boundary_bounds[1], boundary_bounds[3]]),
    ])
    all_lons = pd.concat([
        match_df["well_lon"], distinct_cells["grid_lon"],
        pd.Series([boundary_bounds[0], boundary_bounds[2]]),
    ])
    lat_min, lat_max = all_lats.min(), all_lats.max()
    lon_min, lon_max = all_lons.min(), all_lons.max()
    lat_buf = (lat_max - lat_min) * 0.15 or 0.05
    lon_buf = (lon_max - lon_min) * 0.15 or 0.05
    lat_min, lat_max = lat_min - lat_buf, lat_max + lat_buf
    lon_min, lon_max = lon_min - lon_buf, lon_max + lon_buf

    print(f"Fetching Esri World Imagery tiles for bbox ({lat_min:.3f}, {lon_min:.3f}) to ({lat_max:.3f}, {lon_max:.3f})...")
    basemap_img, extent_3857 = fetch_esri_basemap(lat_min, lat_max, lon_min, lon_max, zoom=11)

    xmin, ymin = _to_3857(lon_min, lat_min)
    xmax, ymax = _to_3857(lon_max, lat_max)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(basemap_img, extent=extent_3857, zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # district boundary outline -- on top of the basemap, underneath markers
    thanjavur_boundary.plot(ax=ax, facecolor="none", edgecolor="yellow",
                             linewidth=2, zorder=1)

    # connecting lines: well -> its matched grid cell
    for row in wells_gdf.itertuples():
        cell_row = cells_gdf[cells_gdf["grid_id"] == row.grid_id].iloc[0]
        ax.plot(
            [row.geometry.x, cell_row.geometry.x],
            [row.geometry.y, cell_row.geometry.y],
            color="white", linewidth=0.8, alpha=0.85, zorder=2,
            path_effects=None,
        )

    wells_gdf.plot(ax=ax, color="#1f77ff", markersize=55, marker="o",
                    edgecolor="white", linewidth=0.8, zorder=3, label="_nolegend_")
    cells_gdf.plot(ax=ax, color="#ff5a1f", markersize=140, marker="s",
                    edgecolor="white", linewidth=1.0, zorder=4, label="_nolegend_")

    for row in wells_gdf.itertuples():
        ax.annotate(row.well_name, (row.geometry.x, row.geometry.y),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=6.5, color="white",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    for row in cells_gdf.itertuples():
        ax.annotate(row.grid_id, (row.geometry.x, row.geometry.y),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=10, fontweight="bold", color="#ff5a1f",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])

    ax.set_axis_off()
    ax.set_title("Thanjavur District: Well-to-Rainfall-Grid Spatial Matching",
                  fontsize=15, fontweight="bold", pad=14)

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77ff",
               markeredgecolor="white", markersize=10, label=f"CGWB well (n={len(wells_gdf)})"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#ff5a1f",
               markeredgecolor="white", markersize=11, label=f"IMD rainfall grid cell (n={len(cells_gdf)})"),
        Line2D([0], [0], color="white", linewidth=1.2, label="nearest-cell match"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", framealpha=0.85, fontsize=9)

    scalebar = ScaleBar(1, units="m", location="lower right", box_alpha=0.7, font_properties={"size": 9})
    ax.add_artist(scalebar)

    # north arrow
    ax.annotate("N", xy=(0.96, 0.88), xytext=(0.96, 0.80), xycoords="axes fraction",
                fontsize=13, fontweight="bold", color="white", ha="center",
                arrowprops=dict(facecolor="white", edgecolor="white", width=3, headwidth=10, headlength=10),
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    fig.tight_layout()
    fig.savefig(MAP_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved map: {MAP_OUT}")
    print(f"Saved match table: {CSV_OUT}")
    print(f"\nThanjavur wells: {len(match_df)}")
    print(f"Distinct grid cells used: {len(distinct_cells)}")
    print("\nFull well -> grid match table:")
    print(match_df.to_string(index=False))


if __name__ == "__main__":
    main()
