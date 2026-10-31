"""
Excel export: all 53 wells with their matched IMD rainfall grid cell.

Read-only. Does NOT touch merge.py, merged_final.csv, or any existing
map file. Imports merge.py's own haversine_km() / nearest_cell() so the
matching logic is identical to Phase B and the existing maps -- not a
separately-written copy. Grid IDs (G1-G18) use the exact same sort
order as map_full_study_area.py (grid_lat descending, then grid_lon
ascending) so IDs match the existing maps exactly, not a fresh
numbering.

Output:
    figures/well_grid_matching_all53.xlsx
"""

import os
import sys

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

sys.path.insert(0, DATA_DIR)
from merge import haversine_km, nearest_cell  # noqa: E402

CGWB_PATH = os.path.join(DATA_DIR, "cgwb_thanjavur_thiruvarur_nagapattinam.csv")
RAIN_PATH = os.path.join(DATA_DIR, "imd_rain_thanjavur_thiruvarur_nagapattinam.csv")
OUT_PATH = os.path.join(FIGURES_DIR, "well_grid_matching_all53.xlsx")

COLUMNS = [
    "Well Name", "District", "Well Latitude", "Well Longitude",
    "Matched Grid ID", "Grid Latitude", "Grid Longitude", "Distance to Grid (km)",
]

HEADER_FILL = PatternFill(start_color="FFD9E6F2", end_color="FFD9E6F2", fill_type="solid")
HEADER_FONT = Font(name="Arial", bold=True, size=11)
BODY_FONT = Font(name="Arial", size=10)
SECTION_FONT = Font(name="Arial", bold=True, size=11)
NOTE_FONT = Font(name="Arial", italic=True, size=9, color="FF555555")


def build_match_table():
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
        matches.append((
            well.station_name, well.district_name, well.latitude, well.longitude,
            cell_lat, cell_lon, dist_km,
        ))

    match_df = pd.DataFrame(matches, columns=[
        "well_name", "district", "well_lat", "well_lon", "grid_lat", "grid_lon", "distance_km",
    ])

    # same grid_id assignment order as map_full_study_area.py -- IDs must match exactly
    all_cells = (
        rain[["grid_lat", "grid_long"]]
        .drop_duplicates()
        .rename(columns={"grid_long": "grid_lon"})
        .sort_values(["grid_lat", "grid_lon"], ascending=[False, True])
        .reset_index(drop=True)
    )
    all_cells["grid_id"] = ["G" + str(i + 1) for i in range(len(all_cells))]

    match_df = match_df.merge(all_cells, on=["grid_lat", "grid_lon"], how="left")
    match_df = match_df.sort_values(["district", "well_name"]).reset_index(drop=True)

    return match_df


def build_summary(match_df):
    summary = (
        match_df.groupby("district")
        .agg(well_count=("well_name", "nunique"), grid_cells_used=("grid_id", "nunique"))
        .reset_index()
        .sort_values("district")
    )
    return summary


def autofit_columns(ws, min_width=10, max_width=40, padding=2):
    for col_cells in ws.columns:
        col_letter = get_column_letter(col_cells[0].column)
        max_len = max((len(str(c.value)) for c in col_cells if c.value is not None), default=0)
        ws.column_dimensions[col_letter].width = max(min_width, min(max_width, max_len + padding))


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    match_df = build_match_table()
    summary = build_summary(match_df)

    print("Wells per district / distinct grid cells used per district:")
    for row in summary.itertuples():
        print(f"  {row.district}: {row.well_count} wells, {row.grid_cells_used} distinct grid cells")

    wb = Workbook()
    ws = wb.active
    ws.title = "Well-Grid Matching"

    ws.append(COLUMNS)
    for cell in ws[1]:
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row in match_df.itertuples(index=False):
        ws.append([
            row.well_name, row.district, round(row.well_lat, 5), round(row.well_lon, 5),
            row.grid_id, round(row.grid_lat, 4), round(row.grid_lon, 4), round(row.distance_km, 2),
        ])

    for r in range(2, ws.max_row + 1):
        for c in range(1, len(COLUMNS) + 1):
            ws.cell(row=r, column=c).font = BODY_FONT

    ws.freeze_panes = "A2"

    # --- summary section, a few rows below the main table ---
    start_row = ws.max_row + 3

    ws.cell(row=start_row, column=1, value="Summary: wells and distinct grid cells used per district").font = SECTION_FONT

    summary_header_row = start_row + 1
    summary_cols = ["District", "Well Count", "Distinct Grid Cells Used"]
    for c, col_name in enumerate(summary_cols, start=1):
        cell = ws.cell(row=summary_header_row, column=c, value=col_name)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for i, row in enumerate(summary.itertuples(index=False), start=1):
        r = summary_header_row + i
        ws.cell(row=r, column=1, value=row.district).font = BODY_FONT
        ws.cell(row=r, column=2, value=row.well_count).font = BODY_FONT
        ws.cell(row=r, column=3, value=row.grid_cells_used).font = BODY_FONT

    total_row = summary_header_row + len(summary) + 1
    ws.cell(row=total_row, column=1, value="Total").font = HEADER_FONT
    ws.cell(row=total_row, column=2, value=int(summary["well_count"].sum())).font = HEADER_FONT
    ws.cell(row=total_row, column=3, value=int(match_df["grid_id"].nunique())).font = HEADER_FONT

    note_row = total_row + 2
    ws.cell(row=note_row, column=1,
            value="Note: Grid IDs (G1-G18) match the numbering used in figures/full_study_area_map.png "
                  "and the interactive/Thanjavur maps -- same sort order (grid_lat descending, grid_lon "
                  "ascending), same nearest_cell()/haversine_km() logic from data/merge.py.").font = NOTE_FONT
    ws.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=len(COLUMNS))
    ws.cell(row=note_row, column=1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[note_row].height = 30  # openpyxl doesn't auto-size wrapped rows

    autofit_columns(ws)

    wb.save(OUT_PATH)

    print(f"\nSaved: {OUT_PATH}")
    print(f"Total rows: {len(match_df)}")
    print(f"Unique well names: {match_df['well_name'].nunique()}")
    print(f"Duplicate well rows: {len(match_df) - match_df['well_name'].nunique()}")

    keelvalur = match_df[match_df["well_name"] == "Keelvalur"]
    print("\nSpot check -- Keelvalur:")
    print(keelvalur.to_string(index=False))


if __name__ == "__main__":
    main()
