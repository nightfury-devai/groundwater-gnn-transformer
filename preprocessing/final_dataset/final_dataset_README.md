```
Final GAP dataset (gap-tolerant windows), Cauvery delta groundwater forecasting

Window = last 8 readings of one well; target = depth at the next reading.

Source files: preprocessing/c3_windows_gap.npz (windows, split) and preprocessing/c2_normalised.csv (raw values, dates). Nothing is filled or interpolated; displayed numbers are rounded to 3 decimals, stored values are full precision.

Split rule (chronological, by TARGET date, no shuffling, cutoffs from c4_split_dates.json):
  train: target_date <= 2022-09-30;  val: 2022-09-30 < target_date <= 2023-12-31;  test: target_date > 2023-12-31.
Windows per split: train 581, val 147, test 112, total 840 (from 42 wells in 3 districts).

Sheets:
  README        this text.
  Summary       one row per window (840 rows).
  Windows_long  one row per window position (840 x 8 = 6,720 rows): the model input.
  Split_counts  windows and wells per split, per district, and per split and district.

Summary columns:
  window_id             1..840, windows ordered by well and then by time.
  well_id               CGWB station name of the well.
  district              Thanjavur, Nagapattinam or Thiruvarur.
  split                 train, val or test (rule above).
  last_input_date       date of the 8th (most recent) input reading.
  target_date           date of the next reading, whose depth is the target.
  target_gap_quarters   whole quarters from the last input reading to the target reading (1 = next quarter).
  last_input_depth_m    depth to water level at the last input reading (the persistence forecast), metres.
  target_depth_m        depth to water level at the target reading, metres.

Windows_long columns:
  window_id             links to Summary.
  position              1 (oldest) to 8 (most recent input reading).
  reading_date          date of the reading at this position.
  gap_before_quarters   whole quarters between the previous reading of the well and this one; empty for position 1 when the window starts at the first reading of the well (nothing before it).
  rainfall_180d_avg_mm  180-day trailing mean of daily IMD rainfall, mm/day.
  tmax_90d_avg_c        90-day trailing mean of maximum temperature, deg C.
  tmin_90d_avg_c        90-day trailing mean of minimum temperature, deg C.
  humidity_90d_avg_pct  90-day trailing mean of relative humidity (NASA POWER), %.
  et_90d_sum_mm         90-day trailing sum of evapotranspiration (NASA POWER), mm.
  depth_m               depth to water level at this reading, metres.
  z_<feature>           the same six values as z-scores: (value - mean) / std, with mean and std from the readings dated on or before the train cutoff only (preprocessing/c2_scaler.json). Depth statistics are shared with the target.

Negative depths: three Allankottai readings are below zero in the source data and are kept as they are.
```
