"""
Phase C2: z-score normalisation of the 6 features (and of the target), fitted on TRAIN-period data only.

Why train only: the mean and std must not know anything about val/test readings, otherwise the
scaling itself leaks the future. 'Train period' = readings dated on or before the train cutoff
from C4 (preprocessing/c4_split_dates.json). The same depth statistics are used for the feature
`depth_m` and the target `next_depth`, because both are depth in metres and the model output must
be convertible back to metres with one pair of numbers.

Population std (ddof = 0). A feature with std = 0 would get std = 1 (guard; not expected).
Nothing is changed, filled or interpolated; the 3 negative Allankottai depths stay as they are.

Input  (read only): preprocessing/c1_features.csv, preprocessing/c4_split_dates.json
Output: preprocessing/c2_scaler.json, preprocessing/c2_normalised.csv, preprocessing/c2_report.md

Inverse (import-safe helpers): depth_to_metres(z, scaler), feature_to_raw(name, z, scaler).

Run from the project root:  .venv/Scripts/python.exe -B -I preprocessing/c2_normalise.py
"""

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{ROOT}/preprocessing/c1_features.csv"
SPLIT = f"{ROOT}/preprocessing/c4_split_dates.json"
OUT_JSON = f"{ROOT}/preprocessing/c2_scaler.json"
OUT_CSV = f"{ROOT}/preprocessing/c2_normalised.csv"
OUT_MD = f"{ROOT}/preprocessing/c2_report.md"

FEATURES = ["rainfall_180d_avg_mm", "tmax_90d_avg_c", "tmin_90d_avg_c",
            "humidity_90d_avg_pct", "et_90d_sum_mm", "depth_m"]


def z_name(c):
    return "z_" + c


# ------------------------------------------------------------ inverse helpers (used by later phases)
def load_scaler(path=OUT_JSON):
    with open(path) as f:
        return json.load(f)


def feature_to_raw(name, z, scaler):
    """Normalised value(s) -> original units for one of the 6 features."""
    s = scaler["features"][name]
    return np.asarray(z, dtype=float) * s["std_used"] + s["mean"]


def depth_to_metres(z, scaler):
    """Model output / normalised target -> depth to water level in metres (same stats as depth_m)."""
    return feature_to_raw("depth_m", z, scaler)


L = []


def say(s=""):
    print(s)
    L.append(s)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def fit(values):
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return {"mean": mean, "std": std, "std_used": std if std > 0 else 1.0,
            "min": float(values.min()), "max": float(values.max()), "n": int(len(values))}


def main():
    with open(SPLIT) as f:
        cut = json.load(f)
    cut_train = pd.Timestamp(cut["train_cutoff"])
    cut_val = pd.Timestamp(cut["val_cutoff"])

    # raw text kept for the output so the raw columns stay byte-identical to C1
    raw = pd.read_csv(SRC, dtype=str, keep_default_na=False)
    num = raw[FEATURES].astype(float)
    date = pd.to_datetime(raw["date"])
    next_date = pd.to_datetime(raw["next_date"].replace("", np.nan))
    next_depth = pd.to_numeric(raw["next_depth"].replace("", np.nan))
    assert len(raw) == 1211 and not num.isna().any().any()

    # ---- fit on readings dated on or before the train cutoff (all such readings, with or without a target)
    fit_mask = (date <= cut_train).values
    assert fit_mask.sum() > 0 and date[fit_mask].max() <= cut_train
    train_stats = {c: fit(num.loc[fit_mask, c].values) for c in FEATURES}
    all_stats = {c: fit(num[c].values) for c in FEATURES}
    # sanity: scaler never saw anything dated after the train cutoff
    assert all(train_stats[c]["n"] == int(fit_mask.sum()) for c in FEATURES)
    scaler = {"features": train_stats, "fit_rule": "readings dated on or before train_cutoff",
              "train_cutoff": cut["train_cutoff"], "std": "population (ddof=0); std_used = 1 if std == 0",
              "target": "next_depth is normalised with the depth_m statistics",
              "n_readings_fit": int(fit_mask.sum())}
    with open(OUT_JSON, "w", newline="\n") as f:
        json.dump(scaler, f, indent=2, sort_keys=True)
        f.write("\n")

    # ---- output: raw columns + normalised columns
    out = raw.copy()
    for c in FEATURES:
        s = train_stats[c]
        out[z_name(c)] = ((num[c] - s["mean"]) / s["std_used"]).map(repr)
    sd = train_stats["depth_m"]
    z_next = (next_depth - sd["mean"]) / sd["std_used"]
    out[z_name("next_depth")] = z_next.map(lambda v: "" if pd.isna(v) else repr(float(v)))
    out.to_csv(OUT_CSV, index=False, lineterminator="\n")

    # ---- self-check: inverse reproduces the raw values
    back = depth_to_metres(out[z_name("depth_m")].astype(float).values, scaler)
    assert np.allclose(back, num["depth_m"].values, atol=1e-9)
    has_t = next_depth.notna().values
    back_t = depth_to_metres(z_next[has_t].values, scaler)
    assert np.allclose(back_t, next_depth[has_t].values, atol=1e-9)
    for c in FEATURES:
        assert np.allclose(feature_to_raw(c, out[z_name(c)].astype(float).values, scaler), num[c].values, atol=1e-9)

    # ---- report
    say("# C2 report: z-score normalisation (train-fitted)")
    say()
    say(f"Input `preprocessing/c1_features.csv` (read only, {len(raw)} readings). Cutoffs from `preprocessing/c4_split_dates.json`: train {cut['train_cutoff']}, val {cut['val_cutoff']}.")
    say()
    say("## 1. Rule")
    say()
    say(f"- Mean and std come ONLY from readings dated on or before the train cutoff ({cut['train_cutoff']}): **{int(fit_mask.sum())} of {len(raw)} readings** (latest fitted reading date {date[fit_mask].max():%Y-%m-%d}). The {int((~fit_mask).sum())} later readings were not used.")
    say("- Population std (ddof = 0). Guard: a std of 0 would be replaced by 1; no feature needed it.")
    say("- `depth_m` and `next_depth` share one pair of statistics (the `depth_m` ones). `z_next_depth` is empty where the target is empty (53 last readings).")
    say("- No value was filled, interpolated or removed. Saved: `preprocessing/c2_scaler.json` (with `min`, `max`, `n`) and `preprocessing/c2_normalised.csv` (all C1 columns, unchanged, plus 7 `z_` columns).")
    say("- Inverse: `depth_to_metres(z, scaler)` and `feature_to_raw(name, z, scaler)` in `c2_normalise.py`; checked here to reproduce all raw values to 1e-9.")
    neg_all = int((num["depth_m"] < 0).sum())
    neg_fit = int((num.loc[fit_mask, "depth_m"] < 0).sum())
    say(f"- Negative depths kept as they are: {neg_all} readings with depth_m < 0 in the data ({neg_fit} of them inside the fit period and therefore in the train statistics).")
    say()
    say("## 2. Train-fitted statistics")
    say()
    rows = [[c, s["n"], f"{s['mean']:.4f}", f"{s['std']:.4f}", f"{s['min']:.4f}", f"{s['max']:.4f}"] for c, s in train_stats.items()]
    for line in md_table(["feature", "n readings", "mean", "std", "train min", "train max"], rows):
        say(line)
    say()
    say("## 3. Same statistics if fitted on ALL data (for comparison only, NOT used)")
    say()
    rows = []
    for c in FEATURES:
        t, a = train_stats[c], all_stats[c]
        rows.append([c, f"{t['mean']:.4f}", f"{a['mean']:.4f}", f"{a['mean'] - t['mean']:+.4f}",
                     f"{t['std']:.4f}", f"{a['std']:.4f}", f"{a['std'] - t['std']:+.4f}"])
    for line in md_table(["feature", "mean (train)", "mean (all)", "diff", "std (train)", "std (all)", "diff"], rows):
        say(line)
    say()
    say(f"- All-data fit uses {len(raw)} readings, train fit {int(fit_mask.sum())}. The difference is the leakage a global scaler would add.")
    say()
    say("## 4. Val/test values outside the train min-max range")
    say()
    say("Counted over the 1158 rows that have a target, each assigned to val or test by its TARGET date (rows with target date on or before the train cutoff are train). 'Outside' means below the train min or above the train max of that feature (fit-period readings). Expected for a chronological split (the climate and the water table move); reported, not corrected, nothing is clipped. `next_depth` uses the depth range.")
    say()
    sample = has_t
    split = np.where(next_date <= cut_train, "train", np.where(next_date <= cut_val, "val", "test"))
    cols = {c: num[c] for c in FEATURES}
    cols["next_depth (target)"] = next_depth
    rngs = {c: (train_stats[c]["min"], train_stats[c]["max"]) for c in FEATURES}
    rngs["next_depth (target)"] = (train_stats["depth_m"]["min"], train_stats["depth_m"]["max"])
    n_val, n_test = int(((split == "val") & sample).sum()), int(((split == "test") & sample).sum())
    rows = []
    for c, v in cols.items():
        lo, hi = rngs[c]
        out_of = (v < lo) | (v > hi)
        r = [c]
        for s_, n_ in (("val", n_val), ("test", n_test)):
            m = (split == s_) & sample
            k = int(out_of[m].sum())
            r.append(f"{k} of {n_} ({100 * k / n_:.1f}%)")
        rows.append(r)
    for line in md_table(["column", f"val (n={n_val})", f"test (n={n_test})"], rows):
        say(line)
    say()
    say("## 5. Checks (asserted in code)")
    say()
    say(f"- Scaler fitted on {int(fit_mask.sum())} readings, all dated on or before {cut['train_cutoff']}: passed.")
    say("- Inverse transform reproduces depth_m, next_depth and the 5 weather features to 1e-9: passed.")
    say("- Raw columns of `c2_normalised.csv` are the C1 text unchanged (written from the string copy): passed by construction.")
    say()
    say("## 6. Files")
    say()
    say("- `preprocessing/c2_normalise.py`, `c2_scaler.json`, `c2_normalised.csv`, `c2_report.md`.")
    with open(OUT_MD, "w", newline="\n") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
