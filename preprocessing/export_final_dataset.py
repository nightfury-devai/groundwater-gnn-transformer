"""
Human-readable export of the final GAP dataset (read-only on everything else).

Input  (read only): preprocessing/c3_windows_gap.npz, preprocessing/c2_normalised.csv,
                    preprocessing/c4_split_dates.json
Output (new folder preprocessing/final_dataset/):
    final_dataset_gap.xlsx   (README, Summary, Windows_long, Split_counts)
    final_dataset_gap_summary.csv, final_dataset_gap_windows_long.csv, final_dataset_README.md

Raw feature values and the date of every window position are looked up in c2_normalised.csv by
(well, last_input_date): a window is the 8 consecutive readings of that well ending at the last input
reading. The z-values rebuilt from the table are asserted equal to the npz X, so the lookup is verified.
Nothing is changed, filled or rounded in the data: rounding to 3 decimals is a display format only.

Run from the project root:  .venv/Scripts/python.exe -B -I preprocessing/export_final_dataset.py
"""

import datetime as dt
import json
import os

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = f"{ROOT}/preprocessing/c3_windows_gap.npz"
TABLE = f"{ROOT}/preprocessing/c2_normalised.csv"
SPLIT = f"{ROOT}/preprocessing/c4_split_dates.json"
OUT = f"{ROOT}/preprocessing/final_dataset"
LWIN = 8
FEATURES = ["rainfall_180d_avg_mm", "tmax_90d_avg_c", "tmin_90d_avg_c",
            "humidity_90d_avg_pct", "et_90d_sum_mm", "depth_m"]
ZF = ["z_" + c for c in FEATURES]
SPLITS = ["train", "val", "test"]
DISTRICTS = ["Thanjavur", "Nagapattinam", "Thiruvarur"]

os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------ load
z = np.load(NPZ, allow_pickle=False)
with open(SPLIT) as f:
    cut = json.load(f)
N = len(z["y"])
assert z["X"].shape == (N, LWIN, 6) and N == 840

tab = pd.read_csv(TABLE, keep_default_na=False)
tab["date"] = pd.to_datetime(tab["date"])
tab = tab.sort_values(["well_id", "date"], kind="stable").reset_index(drop=True)
for c in ["gap_from_prev_quarters"]:
    tab[c] = pd.to_numeric(tab[c], errors="coerce")
by_well = {w: d.reset_index(drop=True) for w, d in tab.groupby("well_id")}

# ------------------------------------------------------------ Windows_long (one row per window position)
rows = []
for i in range(N):
    d = by_well[str(z["well_id"][i])]
    k = int(np.flatnonzero(d["date"].values == np.datetime64(str(z["last_input_date"][i])))[0])
    blk = d.iloc[k - LWIN + 1:k + 1]
    assert len(blk) == LWIN
    # the table block must be exactly what the npz holds (normalised features and gaps)
    assert np.allclose(blk[ZF].values, z["X"][i], atol=1e-12, rtol=0), f"z mismatch in window {i + 1}"
    gb = blk["gap_from_prev_quarters"].values
    assert np.array_equal(np.isnan(gb), np.isnan(z["G"][i])) and np.allclose(gb[~np.isnan(gb)], z["G"][i][~np.isnan(gb)])
    for p in range(LWIN):
        r = {"window_id": i + 1, "position": p + 1, "reading_date": blk["date"].iloc[p],
             "gap_before_quarters": gb[p]}
        for c in FEATURES:
            r[c] = blk[c].iloc[p]
        for c in ZF:
            r[c] = blk[c].iloc[p]
        rows.append(r)
long_df = pd.DataFrame(rows)
assert len(long_df) == N * LWIN == 6720
# the last position's date must be the window's last input date
assert (long_df[long_df["position"] == LWIN]["reading_date"].dt.strftime("%Y-%m-%d").values == z["last_input_date"]).all()

# ------------------------------------------------------------ Summary (one row per window)
last_depth = long_df[long_df["position"] == LWIN]["depth_m"].values
summary = pd.DataFrame({
    "window_id": np.arange(1, N + 1), "well_id": z["well_id"], "district": z["district"], "split": z["split"],
    "last_input_date": pd.to_datetime(z["last_input_date"]), "target_date": pd.to_datetime(z["target_date"]),
    "target_gap_quarters": z["target_gap"], "last_input_depth_m": last_depth, "target_depth_m": z["y_metres"]})
assert len(summary) == 840
cnt = summary["split"].value_counts()
assert (cnt["train"], cnt["val"], cnt["test"]) == (581, 147, 112)
assert (summary["target_date"] > summary["last_input_date"]).all()

# ------------------------------------------------------------ Split_counts
def count_row(label, m):
    return [label, int(m.sum()), int(summary.loc[m, "well_id"].nunique())]


sc_rows = [["By split", "", ""]]
for s in SPLITS:
    sc_rows.append(count_row(s, summary["split"] == s))
sc_rows.append(count_row("all", summary["split"].notna()))
sc_rows.append(["By district", "", ""])
for dname in DISTRICTS:
    sc_rows.append(count_row(dname, summary["district"] == dname))
sc_rows.append(["By split and district", "", ""])
for s in SPLITS:
    for dname in DISTRICTS:
        sc_rows.append(count_row(f"{s} / {dname}", (summary["split"] == s) & (summary["district"] == dname)))
sc_header = ["group", "windows", "wells"]

# ------------------------------------------------------------ README text
n_s = {s: int(cnt[s]) for s in SPLITS}
readme = [
    "Final GAP dataset (gap-tolerant windows), Cauvery delta groundwater forecasting",
    "",
    "Window = last 8 readings of one well; target = depth at the next reading.",
    "",
    "Source files: preprocessing/c3_windows_gap.npz (windows, split) and preprocessing/c2_normalised.csv (raw values, dates). Nothing is filled or interpolated; displayed numbers are rounded to 3 decimals, stored values are full precision.",
    "",
    "Split rule (chronological, by TARGET date, no shuffling, cutoffs from c4_split_dates.json):",
    f"  train: target_date <= {cut['train_cutoff']};  val: {cut['train_cutoff']} < target_date <= {cut['val_cutoff']};  test: target_date > {cut['val_cutoff']}.",
    f"Windows per split: train {n_s['train']}, val {n_s['val']}, test {n_s['test']}, total {N} (from 42 wells in 3 districts).",
    "",
    "Sheets:",
    "  README        this text.",
    "  Summary       one row per window (840 rows).",
    "  Windows_long  one row per window position (840 x 8 = 6,720 rows): the model input.",
    "  Split_counts  windows and wells per split, per district, and per split and district.",
    "",
    "Summary columns:",
    "  window_id             1..840, windows ordered by well and then by time.",
    "  well_id               CGWB station name of the well.",
    "  district              Thanjavur, Nagapattinam or Thiruvarur.",
    "  split                 train, val or test (rule above).",
    "  last_input_date       date of the 8th (most recent) input reading.",
    "  target_date           date of the next reading, whose depth is the target.",
    "  target_gap_quarters   whole quarters from the last input reading to the target reading (1 = next quarter).",
    "  last_input_depth_m    depth to water level at the last input reading (the persistence forecast), metres.",
    "  target_depth_m        depth to water level at the target reading, metres.",
    "",
    "Windows_long columns:",
    "  window_id             links to Summary.",
    "  position              1 (oldest) to 8 (most recent input reading).",
    "  reading_date          date of the reading at this position.",
    "  gap_before_quarters   whole quarters between the previous reading of the well and this one; empty for position 1 when the window starts at the first reading of the well (nothing before it).",
    "  rainfall_180d_avg_mm  180-day trailing mean of daily IMD rainfall, mm/day.",
    "  tmax_90d_avg_c        90-day trailing mean of maximum temperature, deg C.",
    "  tmin_90d_avg_c        90-day trailing mean of minimum temperature, deg C.",
    "  humidity_90d_avg_pct  90-day trailing mean of relative humidity (NASA POWER), %.",
    "  et_90d_sum_mm         90-day trailing sum of evapotranspiration (NASA POWER), mm.",
    "  depth_m               depth to water level at this reading, metres.",
    "  z_<feature>           the same six values as z-scores: (value - mean) / std, with mean and std from the readings dated on or before the train cutoff only (preprocessing/c2_scaler.json). Depth statistics are shared with the target.",
    "",
    "Negative depths: three Allankottai readings are below zero in the source data and are kept as they are.",
]

# ------------------------------------------------------------ CSV and Markdown
sum_csv = summary.copy()
sum_csv["last_input_date"] = sum_csv["last_input_date"].dt.strftime("%Y-%m-%d")
sum_csv["target_date"] = sum_csv["target_date"].dt.strftime("%Y-%m-%d")
sum_csv.to_csv(f"{OUT}/final_dataset_gap_summary.csv", index=False, lineterminator="\n")
long_csv = long_df.copy()
long_csv["reading_date"] = long_csv["reading_date"].dt.strftime("%Y-%m-%d")
long_csv.to_csv(f"{OUT}/final_dataset_gap_windows_long.csv", index=False, lineterminator="\n")
with open(f"{OUT}/final_dataset_README.md", "w", newline="\n", encoding="utf8") as f:
    f.write("```\n" + "\n".join(readme) + "\n```\n")

# ------------------------------------------------------------ Excel
BOLD = Font(bold=True)


def to_cell(v):
    if isinstance(v, pd.Timestamp):
        return v.date()
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)      # NaN -> empty cell
    if isinstance(v, np.integer):
        return int(v)
    return v


def write_df(ws, df, date_cols=(), int_cols=()):
    ws.append(list(df.columns))
    for rec in df.itertuples(index=False):
        ws.append([to_cell(v) for v in rec])
    for c in ws[1]:
        c.font = BOLD
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for j, name in enumerate(df.columns, start=1):
        col = get_column_letter(j)
        width = max(len(str(name)), 10) + 2
        for cell in ws[col][1:]:
            if name in date_cols:
                cell.number_format = "yyyy-mm-dd"
            elif name in int_cols:
                cell.number_format = "0"
            elif isinstance(cell.value, float):
                cell.number_format = "0.000"          # display only; the stored value keeps full precision
        if name in date_cols:
            width = 13
        ws.column_dimensions[col].width = min(max(width, 11), 30)


wb = Workbook()
wb.properties.creator = "export_final_dataset.py"
wb.properties.created = dt.datetime(2026, 1, 1)       # fixed, so the file metadata does not vary between runs
wb.properties.modified = dt.datetime(2026, 1, 1)

ws = wb.active
ws.title = "README"
for line in readme:
    ws.append([line])
ws["A1"].font = Font(bold=True, size=13)
for r in ws.iter_rows(min_row=1, max_row=ws.max_row):
    r[0].alignment = Alignment(wrap_text=True, vertical="top")
    if r[0].value in ("Sheets:", "Summary columns:", "Windows_long columns:") or str(r[0].value).startswith("Split rule"):
        r[0].font = BOLD
ws.column_dimensions["A"].width = 130

write_df(wb.create_sheet("Summary"), summary, date_cols=("last_input_date", "target_date"),
         int_cols=("window_id", "target_gap_quarters"))
write_df(wb.create_sheet("Windows_long"), long_df, date_cols=("reading_date",),
         int_cols=("window_id", "position", "gap_before_quarters"))

ws = wb.create_sheet("Split_counts")
ws.append(sc_header)
for r in sc_rows:
    ws.append(r)
for c in ws[1]:
    c.font = BOLD
for row in ws.iter_rows(min_row=2):
    if row[1].value == "":
        row[0].font = BOLD
ws.freeze_panes = "A2"
ws.auto_filter.ref = ws.dimensions
ws.column_dimensions["A"].width = 32
ws.column_dimensions["B"].width = 12
ws.column_dimensions["C"].width = 10

XLSX = f"{OUT}/final_dataset_gap.xlsx"
wb.save(XLSX)

# ------------------------------------------------------------ verify by reading the workbook back
chk = pd.read_excel(XLSX, sheet_name=None, engine="openpyxl")
assert list(chk) == ["README", "Summary", "Windows_long", "Split_counts"]
assert len(chk["Summary"]) == 840 and len(chk["Windows_long"]) == 6720
assert chk["Summary"]["split"].value_counts().reindex(SPLITS).tolist() == [581, 147, 112]
assert np.allclose(chk["Windows_long"][ZF].values, long_df[ZF].values, atol=0, rtol=1e-14)   # full precision kept (xlsx stores 15-17 significant digits)
assert np.allclose(chk["Windows_long"][FEATURES].values, long_df[FEATURES].values, atol=0, rtol=1e-14)
assert chk["Windows_long"]["gap_before_quarters"].isna().sum() == int(np.isnan(z["G"][:, 0]).sum())
print(f"Summary rows {len(chk['Summary'])}, Windows_long rows {len(chk['Windows_long'])}, "
      f"splits {n_s['train']}/{n_s['val']}/{n_s['test']}")
print(XLSX)
print(summary.head(10).to_string(index=False))
