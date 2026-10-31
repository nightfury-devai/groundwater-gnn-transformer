"""
Phase D2: haversine distance matrix between the 53 wells.

Node order (critical): graph row i must be the same well as index i anywhere else in the project.
preprocessing/c3_windows_gap.npz stores only a per-window `well_id` (42 wells appear there; wells with
fewer than 8 readings have no window), so it holds no 53-node index. The order is therefore derived with
the SAME rule c3_windows.py uses: c2_normalised.csv sorted by [well_id, date] (stable), wells taken in
order of appearance (groupby sort=False) = ascending well_id string order. The 42 wells in the npz are
then checked to appear in that same relative order.

Coordinates: data/cgwb_thanjavur_thiruvarur_nagapattinam_2013_2025.csv (station_name = well_id),
one latitude/longitude/district per station (asserted). Read only.

Haversine with R = 6371.0088 km (IUGG mean Earth radius): real great-circle distance, not flat-plane,
which matters at this geographic scale (D2 in CLAUDE.md).

Output: graph/distance_matrix.npy (53 x 53, km, float64), graph/well_order.json
Run from the project root:  .venv/Scripts/python.exe -B -I graph/distance_matrix.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CGWB = f"{ROOT}/data/cgwb_thanjavur_thiruvarur_nagapattinam_2013_2025.csv"
C2 = f"{ROOT}/preprocessing/c2_normalised.csv"
NPZ = f"{ROOT}/preprocessing/c3_windows_gap.npz"
OUT_NPY = f"{ROOT}/graph/distance_matrix.npy"
OUT_ORDER = f"{ROOT}/graph/well_order.json"
R_KM = 6371.0088
N_WELLS = 53


def haversine_matrix(lat_deg, lon_deg, radius=R_KM):
    """Pairwise great-circle distances (km) for arrays of latitudes/longitudes in degrees."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    # clip: rounding can push a hair above 1 for antipodal points (not our case) and below 0 never, but be safe
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def well_order():
    """The 53 well ids in the order used by c3_windows.py (see module docstring)."""
    t = pd.read_csv(C2, usecols=["well_id", "date"], dtype=str, keep_default_na=False)
    t = t.sort_values(["well_id", "date"], kind="stable")
    order = list(dict.fromkeys(t["well_id"]))
    if len(order) != N_WELLS:
        raise SystemExit(f"FAIL: expected {N_WELLS} wells in c2_normalised.csv, found {len(order)}")
    npz_wells = list(dict.fromkeys(np.load(NPZ, allow_pickle=False)["well_id"].tolist()))
    pos = [order.index(w) for w in npz_wells]            # raises ValueError if a window well is unknown
    if pos != sorted(pos):
        raise SystemExit("FAIL: the wells in c3_windows_gap.npz are not in the derived node order")
    return order, len(npz_wells)


def main():
    order, n_npz = well_order()
    cg = pd.read_csv(CGWB, dtype={"station_name": str})
    # one coordinate pair and one district per station, otherwise the node position would be ambiguous
    nun = cg.groupby("station_name")[["latitude", "longitude", "district_name"]].nunique()
    if (nun > 1).any().any():
        raise SystemExit("FAIL: a station has more than one latitude/longitude/district")
    info = cg.drop_duplicates("station_name").set_index("station_name")
    missing = [w for w in order if w not in info.index]
    if missing:
        raise SystemExit(f"FAIL: wells without coordinates: {missing}")
    lat = info.loc[order, "latitude"].to_numpy(float)
    lon = info.loc[order, "longitude"].to_numpy(float)
    dist_names = info.loc[order, "district_name"].tolist()
    if np.isnan(lat).any() or np.isnan(lon).any():
        raise SystemExit("FAIL: missing coordinate")

    d = haversine_matrix(lat, lon)
    d = (d + d.T) / 2                                   # remove the last-bit asymmetry of floating point
    np.fill_diagonal(d, 0.0)
    os.makedirs(os.path.dirname(OUT_NPY), exist_ok=True)
    np.save(OUT_NPY, d)
    with open(OUT_ORDER, "w", newline="\n") as f:
        json.dump({"order_rule": "ascending well_id, same as c3_windows.py (groupby sort=False on c2_normalised.csv sorted by well_id, date)",
                   "n": N_WELLS, "earth_radius_km": R_KM,
                   "wells": [{"index": i, "well_id": w, "lat": float(lat[i]), "lon": float(lon[i]), "district": dist_names[i]}
                             for i, w in enumerate(order)]}, f, indent=1)
        f.write("\n")

    off = d[~np.eye(N_WELLS, dtype=bool)]
    ok = True
    for name, cond in [("shape 53x53", d.shape == (N_WELLS, N_WELLS)),
                       ("symmetric", np.array_equal(d, d.T)),
                       ("diagonal = 0", np.all(np.diag(d) == 0)),
                       ("all finite and >= 0", np.isfinite(d).all() and (d >= 0).all()),
                       ("npz wells (%d) in node order" % n_npz, True)]:
        print(f"{'PASS' if cond else 'FAIL'}  {name}")
        ok &= bool(cond)
    print(f"off-diagonal distance (km): min {off.min():.4f}  median {np.median(off):.3f}  max {off.max():.3f}")
    print(f"wrote {OUT_NPY} and {OUT_ORDER}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
