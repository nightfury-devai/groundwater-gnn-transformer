"""
Adds the sheet "Training_data" (first sheet, opened by default) to
preprocessing/final_dataset/final_dataset_gap.xlsx and writes the same table to
preprocessing/final_dataset/training_data_gap.csv. Other sheets and files are not changed
(verified below by comparing every other sheet before and after).

One row per window (840), same order as Summary. Columns: window_id, split, well_id, district,
target_date, then 48 inputs r1_rainfall .. r8_depth (reading 1 = oldest, 8 = most recent; per reading
rainfall, tmax, tmin, humidity, et, depth; z-scores = X from the npz), then target_z (y) and
target_depth_m (y_metres). Rounding to 3 decimals is a display format only.

Run from the project root:  .venv/Scripts/python.exe -B -I preprocessing/add_training_data_sheet.py
"""

import os

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = f"{ROOT}/preprocessing/c3_windows_gap.npz"
XLSX = f"{ROOT}/preprocessing/final_dataset/final_dataset_gap.xlsx"
CSV = f"{ROOT}/preprocessing/final_dataset/training_data_gap.csv"
SHEET = "Training_data"
SHORT = ["rainfall", "tmax", "tmin", "humidity", "et", "depth"]    # order of the 6 features in X
LWIN = 8

z = np.load(NPZ, allow_pickle=False)
N = len(z["y"])
assert z["X"].shape == (N, LWIN, 6) and N == 840

# ---- build the table
cols = {"window_id": np.arange(1, N + 1), "split": z["split"], "well_id": z["well_id"],
        "district": z["district"], "target_date": pd.to_datetime(z["target_date"])}
X = z["X"]
for p in range(LWIN):
    for f, name in enumerate(SHORT):
        cols[f"r{p + 1}_{name}"] = X[:, p, f]
cols["target_z"] = z["y"]
cols["target_depth_m"] = z["y_metres"]
df = pd.DataFrame(cols)
assert df.shape == (840, 55)
vc = df["split"].value_counts()
assert (vc["train"], vc["val"], vc["test"]) == (581, 147, 112)
# every row equals the npz X[i] flattened (position-major, features in the stated order)
flat = df[[f"r{p + 1}_{n}" for p in range(LWIN) for n in SHORT]].values
assert np.array_equal(flat, X.reshape(N, -1))
assert np.array_equal(df["target_z"].values, z["y"]) and np.array_equal(df["target_depth_m"].values, z["y_metres"])

# ---- CSV
out = df.copy()
out["target_date"] = out["target_date"].dt.strftime("%Y-%m-%d")
out.to_csv(CSV, index=False, lineterminator="\n")

# ---- workbook: read the other sheets first so they can be compared afterwards
before = pd.read_excel(XLSX, sheet_name=None, engine="openpyxl")
before.pop(SHEET, None)
wb = load_workbook(XLSX)
if SHEET in wb.sheetnames:                     # re-run: replace the old copy, nothing else
    del wb[SHEET]
ws = wb.create_sheet(SHEET, 0)
ws.append(list(df.columns))
for rec in df.itertuples(index=False):
    ws.append([v.date() if isinstance(v, pd.Timestamp) else (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, np.integer) else v))
               for v in rec])
for c in ws[1]:
    c.font = Font(bold=True)
ws.freeze_panes = "F2"                         # top row and first 5 columns
ws.auto_filter.ref = ws.dimensions
for j, name in enumerate(df.columns, start=1):
    col = get_column_letter(j)
    for cell in ws[col][1:]:
        if name == "target_date":
            cell.number_format = "yyyy-mm-dd"
        elif name == "window_id":
            cell.number_format = "0"
        elif isinstance(cell.value, float):
            cell.number_format = "0.000"       # display only; full precision stays in the cell
    ws.column_dimensions[col].width = {"window_id": 11, "split": 8, "well_id": 18, "district": 14, "target_date": 13}.get(name, max(len(name) + 2, 12))

# open on the new sheet, and only that sheet selected
wb.active = 0
for s in wb.worksheets:
    s.sheet_view.tabSelected = (s.title == SHEET)
wb.save(XLSX)

# ---- verify
after = pd.read_excel(XLSX, sheet_name=None, engine="openpyxl")
assert list(after)[0] == SHEET and list(after)[1:] == list(before)
for name, d in before.items():
    pd.testing.assert_frame_equal(after[name], d)           # other sheets unchanged
t = after[SHEET]
assert t.shape == (840, 55) and list(t.columns) == list(df.columns)
assert t["split"].value_counts().reindex(["train", "val", "test"]).tolist() == [581, 147, 112]
xr = t[[f"r{p + 1}_{n}" for p in range(LWIN) for n in SHORT]].values
assert np.allclose(xr, X.reshape(N, -1), atol=0, rtol=1e-14)
# same order as Summary
sm = after["Summary"]
assert (sm["window_id"].values == t["window_id"].values).all() and (sm["well_id"] == t["well_id"]).all() and (sm["split"] == t["split"]).all()
assert (pd.to_datetime(sm["target_date"]) == pd.to_datetime(t["target_date"])).all()
wb2 = load_workbook(XLSX)
assert wb2.active.title == SHEET and wb2[SHEET].freeze_panes == "F2"
print(f"{SHEET}: {t.shape[0]} rows x {t.shape[1]} columns; sheets: {wb2.sheetnames}")
print(f"splits train/val/test = {vc['train']}/{vc['val']}/{vc['test']}")
pd.set_option("display.width", 250, "display.max_columns", 60)
print(out.head(3).T.to_string())
