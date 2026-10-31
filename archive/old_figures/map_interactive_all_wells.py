"""
Interactive, zoomable web map: all 53 wells + all 18 IMD rainfall grid
cells across Thanjavur, Thiruvarur, and Nagapattinam districts.

Read-only visualization. Does NOT touch merge.py, merged_final.csv, or
the existing static Thanjavur PNG map. Imports merge.py's own
haversine_km() / nearest_cell() so the matching logic here is identical
to what Phase B actually used, not a separately-written copy.

Output:
    figures/interactive_well_grid_map.html -- standalone, self-contained
    (satellite tiles are the only thing fetched at view-time; everything
    else -- markers, lines, popups, legend -- is baked into the file)
"""

import os
import re
import sys

import folium
import geopandas as gpd
import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

sys.path.insert(0, DATA_DIR)
from merge import haversine_km, nearest_cell  # noqa: E402

CGWB_PATH = os.path.join(DATA_DIR, "cgwb_thanjavur_thiruvarur_nagapattinam.csv")
RAIN_PATH = os.path.join(DATA_DIR, "imd_rain_thanjavur_thiruvarur_nagapattinam.csv")
BOUNDARIES_PATH = os.path.join(DATA_DIR, "tn_district_boundaries.gpkg")
OUT_PATH = os.path.join(FIGURES_DIR, "interactive_well_grid_map.html")

DISTRICT_COLORS = {
    "Thanjavur": "#f5e400",     # yellow
    "Nagapattinam": "#00e5ff",  # cyan
    "Thiruvarur": "#ff2fd6",    # magenta
}

ESRI_TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics"


def inline_external_assets(html_path):
    """Folium/branca pull Leaflet, jQuery, Bootstrap, and icon-font CSS
    from public CDNs by default. Fetch each one and inline it directly
    into the HTML so the only network dependency left is the satellite
    tiles themselves -- required for a truly standalone, offline-capable
    single file."""
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    script_pattern = re.compile(r'<script src="(https://[^"]+)"></script>')
    for url in script_pattern.findall(html):
        print(f"Inlining script: {url}")
        content = requests.get(url, timeout=30).text
        html = html.replace(
            f'<script src="{url}"></script>',
            f'<script>\n{content}\n</script>',
        )

    link_pattern = re.compile(r'<link rel="stylesheet" href="(https://[^"]+)"/>')
    for url in link_pattern.findall(html):
        print(f"Inlining stylesheet: {url}")
        content = requests.get(url, timeout=30).text
        html = html.replace(
            f'<link rel="stylesheet" href="{url}"/>',
            f'<style>\n{content}\n</style>',
        )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

LEGEND_HTML = """
<div style="
    position: fixed;
    bottom: 24px; left: 24px; z-index: 9999;
    background: rgba(20, 25, 22, 0.88);
    color: #f0f0f0;
    padding: 12px 16px;
    border-radius: 8px;
    font-family: Arial, sans-serif;
    font-size: 13px;
    line-height: 1.8;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
">
  <div style="font-weight:bold; margin-bottom:4px;">Legend</div>
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#1f77ff;border:1px solid white;margin-right:8px;"></span>CGWB well (n=53)</div>
  <div><span style="display:inline-block;width:11px;height:11px;background:#ff5a1f;border:1px solid white;margin-right:8px;"></span>IMD rainfall grid cell (n=18)</div>
  <div><span style="display:inline-block;width:16px;height:2px;background:white;margin-right:6px;vertical-align:middle;"></span>nearest-cell match</div>
  <div style="margin-top:6px;font-weight:bold;">District boundaries</div>
  <div><span style="display:inline-block;width:16px;height:2px;background:#f5e400;margin-right:6px;vertical-align:middle;"></span>Thanjavur</div>
  <div><span style="display:inline-block;width:16px;height:2px;background:#00e5ff;margin-right:6px;vertical-align:middle;"></span>Nagapattinam</div>
  <div><span style="display:inline-block;width:16px;height:2px;background:#ff2fd6;margin-right:6px;vertical-align:middle;"></span>Thiruvarur</div>
</div>
"""


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    cgwb = pd.read_csv(CGWB_PATH)
    wells = cgwb.drop_duplicates(subset="station_name")[
        ["station_name", "district_name", "latitude", "longitude"]
    ].reset_index(drop=True)

    rain = pd.read_csv(RAIN_PATH)
    rain_cells = rain[["grid_lat", "grid_long"]].drop_duplicates().to_numpy()

    # nearest grid cell + distance per well, using merge.py's own functions
    matches = []
    for well in wells.itertuples(index=False):
        cell_lat, cell_lon = nearest_cell(well.latitude, well.longitude, rain_cells)
        dist_km = haversine_km(well.latitude, well.longitude, cell_lat, cell_lon)
        matches.append((
            well.station_name, well.district_name, well.latitude, well.longitude,
            cell_lat, cell_lon, dist_km,
        ))

    match_df = pd.DataFrame(matches, columns=[
        "well_name", "district", "well_lat", "well_lon", "grid_lat", "grid_lon", "distance_km",
    ])

    # ALL 18 IMD rain grid cells in the region -- not just the ones that
    # happen to be some well's nearest match. Sorted north-to-south then
    # west-to-east for stable IDs across reruns.
    all_cells = (
        rain[["grid_lat", "grid_long"]]
        .drop_duplicates()
        .rename(columns={"grid_long": "grid_lon"})
        .sort_values(["grid_lat", "grid_lon"], ascending=[False, True])
        .reset_index(drop=True)
    )
    all_cells["grid_id"] = ["G" + str(i + 1) for i in range(len(all_cells))]
    matched_cell_count = match_df[["grid_lat", "grid_lon"]].drop_duplicates().shape[0]
    print(f"Total IMD rain grid cells in region: {len(all_cells)}")
    print(f"Of those, actually used as some well's nearest cell: {matched_cell_count}")

    match_df = match_df.merge(all_cells, on=["grid_lat", "grid_lon"], how="left")
    distinct_cells = all_cells

    # wells per grid cell, for grid-cell popups
    wells_per_cell = match_df.groupby("grid_id")["well_name"].apply(list).to_dict()

    # --- Build the map ---
    center_lat = wells["latitude"].mean()
    center_lon = wells["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles=None, control_scale=True)

    folium.TileLayer(
        tiles=ESRI_TILE_URL,
        attr=ESRI_ATTR,
        name="Esri World Imagery",
        overlay=False,
        control=False,
        max_zoom=19,
    ).add_to(m)

    well_layer = folium.FeatureGroup(name="Wells", show=True)
    grid_layer = folium.FeatureGroup(name="Grid cells", show=True)
    line_layer = folium.FeatureGroup(name="Matches", show=True)
    boundary_layer = folium.FeatureGroup(name="District boundaries", show=True)

    # district boundary outlines, one color per district, with a text label
    boundaries = gpd.read_file(BOUNDARIES_PATH)
    for row in boundaries.itertuples():
        color = DISTRICT_COLORS[row.district_name]
        folium.GeoJson(
            data=gpd.GeoSeries([row.geometry], crs=boundaries.crs).to_json(),
            style_function=lambda _, c=color: {
                "fillOpacity": 0, "color": c, "weight": 3, "opacity": 0.95,
            },
            tooltip=row.district_name,
        ).add_to(boundary_layer)

        centroid = row.geometry.centroid
        folium.Marker(
            location=[centroid.y, centroid.x],
            icon=folium.DivIcon(html=f"""
                <div style="
                    color:{color}; font-family:Arial, sans-serif;
                    font-size:15px; font-weight:bold; white-space:nowrap;
                    text-shadow: -1px -1px 0 #000, 1px -1px 0 #000,
                                 -1px 1px 0 #000, 1px 1px 0 #000;
                ">{row.district_name}</div>
            """),
        ).add_to(boundary_layer)

    # connecting lines first, so markers sit visually on top
    for row in match_df.itertuples():
        folium.PolyLine(
            locations=[[row.well_lat, row.well_lon], [row.grid_lat, row.grid_lon]],
            color="white", weight=1.4, opacity=0.85,
        ).add_to(line_layer)

    for row in match_df.itertuples():
        popup_html = (
            f"<b>{row.well_name}</b><br>"
            f"District: {row.district}<br>"
            f"Coordinates: {row.well_lat:.5f}, {row.well_lon:.5f}<br>"
            f"Matched grid cell: {row.grid_id} ({row.distance_km:.2f} km)"
        )
        folium.CircleMarker(
            location=[row.well_lat, row.well_lon],
            radius=6,
            color="white", weight=1.2,
            fill=True, fill_color="#1f77ff", fill_opacity=0.95,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=row.well_name,
        ).add_to(well_layer)

    for row in distinct_cells.itertuples():
        wells_here = wells_per_cell.get(row.grid_id, [])
        wells_list_html = "".join(f"<li>{w}</li>" for w in wells_here)
        popup_html = (
            f"<b>Grid cell {row.grid_id}</b><br>"
            f"Coordinates: {row.grid_lat:.4f}, {row.grid_lon:.4f}<br>"
            f"Nearest-cell for {len(wells_here)} well(s):"
            f"<ul style='margin:4px 0 0 -18px;'>{wells_list_html}</ul>"
        )
        folium.Marker(
            location=[row.grid_lat, row.grid_lon],
            icon=folium.DivIcon(html=f"""
                <div style="
                    width:16px;height:16px;
                    background:#ff5a1f;border:2px solid white;
                    transform: rotate(0deg);
                    box-shadow:0 0 3px rgba(0,0,0,0.6);
                "></div>
            """),
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=row.grid_id,
        ).add_to(grid_layer)

    line_layer.add_to(m)
    boundary_layer.add_to(m)
    well_layer.add_to(m)
    grid_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    m.get_root().html.add_child(folium.Element(LEGEND_HTML))

    # fit bounds to everything (wells + grid cells) so it's all visible on load
    all_lats = pd.concat([match_df["well_lat"], distinct_cells["grid_lat"]])
    all_lons = pd.concat([match_df["well_lon"], distinct_cells["grid_lon"]])
    m.fit_bounds([[all_lats.min(), all_lons.min()], [all_lats.max(), all_lons.max()]])

    title_html = """
    <div style="
        position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
        z-index: 9999; background: rgba(20,25,22,0.88); color: #f5f5f5;
        padding: 8px 20px; border-radius: 8px; font-family: Arial, sans-serif;
        font-size: 16px; font-weight: bold; box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    ">
        Thanjavur &middot; Thiruvarur &middot; Nagapattinam &mdash; Well-to-Rainfall-Grid Spatial Matching
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(OUT_PATH)
    inline_external_assets(OUT_PATH)

    print(f"Saved: {OUT_PATH}")
    print(f"Total wells plotted: {len(match_df)}")
    print(f"Total distinct grid cells plotted: {len(distinct_cells)}")
    print(f"\nWells per district:\n{wells['district_name'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
