# Spatio-Temporal GNN-Transformer Framework for Groundwater Level Forecasting in the Cauvery Delta Alluvial Aquifer, India

> **UPDATED 2026-10-13.** The working dataset is the EXTENDED 2013-2025 data (merged_final_2013_2025.csv, 1,232 rows). Where an older section below still says 838 rows, 2013-2021, or a 90-day rainfall SUM, the section 'CURRENT STATE (2026-10-13)' at the end of this file wins.

## Project Summary
Undergraduate research project targeting a Scopus-indexed journal (Scientific Reports / Earth Science Informatics / MDPI Water / Journal of Hydroinformatics). Hybrid GNN (spatial) + Transformer (temporal) model to forecast groundwater levels across a well network in Tamil Nadu, India.

**Quality bar:** Must match Hua et al. 2025 (AIP Advances) standard — real equations, ablation studies, robustness experiments, multiple baselines, proper metrics tables.

**Accuracy goal:** match or exceed Hua et al. 2025's benchmark — R² > 0.92, MAE reduction of 12.4%, RMSE reduction of 10.7% vs. their baselines. Internal target, not a promised result — report the actual honest number in the paper regardless of outcome.

## Division of Labor
- **Toothless (human):** research, understanding, architecture decisions, review, final say on any judgment call.
- **Claude (chat):** teaches concepts, reviews Claude Code's plans/output, catches bugs, drafts prompts, updates this file — does NOT lock in judgment calls without the user's explicit agreement first.
- **Claude Code:** all implementation. Toothless runs commands locally and reports back errors/results.

## Advisor Confirmation (NIT faculty)
Explicitly confirmed sufficient: CGWB, IMD rainfall, IMD temperature, humidity, evapotranspiration. No further datasets requested. Cleared to proceed to model architecture.

## Novelty Statement (locked, final)
"We are the first to apply spatio-temporal GNN-Transformer modeling to alluvial aquifer dynamics in the Cauvery delta, using CGWB-IMD fusion to capture monsoon-recharge-driven groundwater depletion in a physically connected rural aquifer system."

Confirmed across 25 peer-reviewed papers (2023–2026), 13 read in full. Only close precedent is Hua et al. 2025 (California, 8 wells, well-data only, different region — no exogenous weather fusion). Re-checked fresh (Aug 2026): no new paper found combining GNN+Transformer for Indian groundwater — gap still holds.

## Study Region (locked)
Thanjavur + Thiruvarur + Nagapattinam districts, Tamil Nadu. CGWB classifies these as ONE continuous alluvial Cauvery-delta aquifer — hydraulically connected via river-deposited sand/silt/clay, giving physical justification for GNN spatial modeling. Hard-rock regions (e.g. most of Trichy) ruled out — water sits in disconnected fractures there.

Real-world motivation: seawater/salinity intrusion from the Bay of Bengal + agricultural over-extraction. Confirmed via real CGWB data analysis: **Thanjavur is classified over-exploited, Thiruvarur semi-critical (70-90% extraction)** — same risk categories locked into Phase O. **Nagapattinam is NOT simply "over-exploited"** — it is excluded from CGWB's standard district assessment entirely, because all its assessed groundwater units are saline and unusable (100% salinity per hydrologist Dr. S. Janakarajan, SaciWATERs). This is a stronger, more specific motivation than "over-exploited" and lines up precisely with this project's salinity-intrusion framing. Source: thenewsminute.com CGWB data investigation, March 2025. **Data note:** Nagapattinam district was administratively bifurcated into Nagapattinam + Mayiladuthurai in 2020 — worth verifying which district boundaries the downloaded CGWB well data (2013-2021, spanning this change) actually used.

**Well count: CONFIRMED 53** (Thanjavur 26, Nagapattinam 16, Thiruvarur 11) — exceeds Hua et al. 2025's 8 wells by a wide margin.

## Datasets — 4, all confirmed, all advisor-approved
- **CGWB** — target variable, depth-to-water-level. DONE.
- **IMD rainfall** — 0.25° grid. DONE.
- **IMD temperature** — 1° grid. DONE.
- **Humidity + evapotranspiration** (NASA POWER, point-based per well). DONE.

## Real Confirmed CGWB Numbers
- ORIGINAL CORE: 838 rows, 53 wells (Thanjavur 26, Nagapattinam 16, Thiruvarur 11), 2013-01-01 to 2021-01-09, 0 missing values.
- EXTENDED (2026-10-13): +394 readings from the CGWB portal 2021-2025 file. Total 1,232 rows, 53 wells, 2013-01-01 to 2024-11-30. See CURRENT STATE.
- `station_name` confirmed safe as unique well_id.
- Readings per well uneven (original 838 rows): min 1, median 15, max 29. 42 of 53 wells have 9+ readings. NOTE: 9+ readings does not mean 9 consecutive quarters, see 'Window definition (OPEN)' in CURRENT STATE.
- **20 well-quarter combinations have 2 readings in the same quarter** (e.g. well Keelvalur: 2019-01-28 and 2019-01-29, both Q1 2019). **Rule (locked): keep the most recent reading per quarter** — not averaged, not interpolated.

## Real Confirmed IMD Numbers
- Rainfall: 18 grid cells, 59,166 rows, 2013-01-01 to 2021-12-31, 0 missing.
- Temperature: 9 clean grid cells (after removing fill values), 29,583 rows. tmax range 23.6–42.2°C, tmin range 14.8–30.1°C.
- 2 real bugs found and fixed during build: (1) merging rain+temp by exact coordinates left 94% of temperature NaN — fixed by matching each independently to nearest well. (2) IMD's temperature fill code (`99.9`) was being read as real data — caught via implausible averages, filtered out.

## Real Confirmed Humidity + ET Numbers (NASA POWER)
- 53 wells, 174,211 rows, 2013-01-01 to 2021-12-31, 0 missing.
- Humidity: 35.5%–95.4%. Evapotranspiration converted from raw energy units (MJ/m²/day) to mm/day (÷2.45) — confirmed via API metadata, not assumed. Final range: 0.0–3.31 mm/day.

## Real Confirmed Merge Numbers (ORIGINAL data/merged_final.csv, now archived; WORKING FILE = data/merged_final_2013_2025.csv)
- 838 rows, 0 missing values. Each well matched to nearest rain cell and nearest temp cell independently; humidity/ET joined directly by well_id.
- Feature recipe (FINAL): rainfall = 180-day trailing AVERAGE; tmax, tmin, humidity = 90-day trailing average; ET = 90-day trailing SUM. (The original merge used a 90-day rainfall SUM; superseded.)
- 51 rows have an incomplete 90-day window — verified: all are readings from 2013-01-01/01-04, the very start of the data, so no prior 90 days could exist.
- Physical plausibility confirmed: well Aduthurai shows depth rising after high rainfall (4.50m in low-rain period vs 3.22m post-monsoon).
- Columns (current): `[well_id, date, rainfall_180d_avg_mm, tmax_90d_avg_c, tmin_90d_avg_c, humidity_90d_avg_pct, et_90d_sum_mm, depth_to_water_level_m]`.
- **Still to apply in Phase C:** the duplicate-quarter dedup rule above — locked but not yet coded.

## Full Architecture (build in this order — do not skip ahead)

### Phase A: Data Acquisition — DONE
**What:** identifying and collecting every raw data source the model needs, before any processing.
**Why a separate phase:** the target (groundwater level) and its drivers (rainfall, temperature, humidity, ET) live in entirely different agency systems (CGWB, IMD, NASA) — this phase is purely sourcing/downloading, no processing yet.
**Done:** CGWB (target, 53 wells, 838 readings, 2013-2021), IMD rain (18 cells) + temp (9 cells), NASA POWER humidity+ET (per-well, point-based). See real numbers above.
**Paper framing:** "Data Sources" paragraph — 4 datasets, 3 agencies, justify each source's relevance (target vs. drivers).

### Phase B: Dataset Integration — DONE
**What:** combining 4 separately-sourced files into 1 table where every row has everything needed for one well at one point in time.
**Why it's a real technical step, not just concatenation:** the 4 sources don't share structure — CGWB is irregular point readings, IMD rain/temp sit on 2 different fixed grids not aligned to wells, NASA POWER is continuous daily data at well points. Two distinct problems solved:
- **Space:** wells don't sit on grid points → nearest-neighbor spatial matching, each well assigned its closest rain cell and closest temp cell independently (grids don't align with each other).
- **Time:** one day's weather doesn't explain a reading taken months later → 90-day trailing window per reading (sum for rainfall/ET, average for temp/humidity), turning raw daily data into "recent conditions leading up to this reading."
**Done:** 838 merged rows, 0 missing, verified via real physical check (well Aduthurai: 4.50m depth in low-rain period vs 3.22m post-monsoon — confirms real physical signal, not noise). See real numbers above.
**Paper framing:** "Data Fusion / Integration" paragraph — nearest-neighbor spatial join + 90-day trailing-window temporal aggregation is a legitimate, citable methodological choice.

**Phase B Formulas (SUPERSEDED — see update below. Original IMD-based sum formula kept here for history):**

Original sum aggregation (rainfall, evapotranspiration — these accumulate over time):
```
R_i,t = sum_{d=t-89}^{t} r_i,d
```
LaTeX: `R_{i,t} = \sum_{d=t-89}^{t} r_{i,d}`

Average aggregation (temperature, humidity — these are conditions, not accumulations):
```
T_i,t = (1/90) * sum_{d=t-89}^{t} tau_i,d
```
LaTeX: `T_{i,t} = \frac{1}{90}\sum_{d=t-89}^{t} \tau_{i,d}`

**FINAL CONFIRMED DECISION on rainfall (supersedes the MSWEP-switch discussion below — advisor confirmed, keeping IMD):**
- Dataset: IMD rainfall — CONFIRMED KEPT, not switched to MSWEP. Backed by literature: Sundaram & Radhakrishnan (2025), J. Earth System Science, evaluated APHRODITE/IMD/ERA5/CHIRPS against real gauge data specifically in the Cauvery Delta (1986-2005) and found IMD "moderate reliability" and explicitly "suitable for operational use." APHRODITE ranked best but is ruled OUT for this project — confirmed via its own official readme (V1801R1, current APHRODITE-2 product) that it only covers 1998-2015, cannot reach 2021.
- Window size: CHANGED from 90 days to 180 days (approximately 6 months), per advisor. Exact day-count (180 flat vs. calendar 6-month, which varies 181-184 days) not yet clarified with advisor — using 180 days as the working assumption, flag if advisor means calendar months instead.
- Aggregation: CHANGED from SUM to AVERAGE for rainfall specifically, per advisor. Rainfall now grouped with temperature/humidity as an averaged "condition" feature rather than a summed accumulation.
- New formula (180-day trailing average):
  ```
  P_i,t = (1/180) * sum_{d=t-179}^{t} p_i,d
  ```
  LaTeX: `P_{i,t} = \frac{1}{180}\sum_{d=t-179}^{t} p_{i,d}`
  Column name changes from `rainfall_90d_sum_mm` to `rainfall_180d_avg_mm`.
- RESOLVED (was open, now confirmed): the 180-day/average change applies ONLY to rainfall, per advisor's own words directly quoted — "average daily rainfall for 6 months." Temperature, humidity stay 90-day average (unchanged, as before). Evapotranspiration (ET) stays 90-day SUM (unchanged, as before) — ET's own physical justification (accumulates over time, same as rainfall used to) still holds even though rainfall's aggregation changed.
- OPEN QUESTION still unresolved: does the window-size change (90→180 days) apply project-wide (i.e., does the Methodology/architecture's "8-quarter window" language also need revisiting), or just to this one rainfall feature's own internal aggregation? These are two different kinds of "window" (feature engineering window vs. model input sequence length) — don't conflate them without checking.
- Downstream impact: the already-drafted "Study Area and Data" paper section (1.2 Rainfall, 1.3 Data Fusion) describes IMD rainfall with 90-day SUM aggregation — needs updating to 180-day AVERAGE once Phase B is rebuilt with the new logic.
- merge.py has ALREADY been updated and re-run with this change (180-day average rainfall) — verified: 838 rows preserved, new column `rainfall_180d_avg_mm` ranges 0-9.69 (real, sensible daily-average range), temp/humidity/ET columns confirmed unchanged. This is DONE, not just planned.
- Nearest-cell spatial matching logic (Step 1 of Phase B) does NOT change from this — same haversine-distance method, same IMD 0.25 deg grid. (But see NEW PENDING DECISION below — a separate, bigger change to this same step is under discussion.)

**Literature survey on dataset choice — full findings, to be added as real citations in the paper's Study Area/Data section and Reference list (NOT YET ADDED to the actual paper draft — still pending):**
1. Sundaram & Radhakrishnan (2025). "Evaluating the accuracy and applicability of precipitation datasets for sustainable agro-hydrological and climatic planning in Tamil Nadu's Cauvery Delta." Journal of Earth System Science. [https://link.springer.com/article/10.1007/s12040-025-02690-3] — the main IMD-justification paper (see above).
2. HESS (2025). "Evaluation of remote-sensing- and reanalysis-based precipitation products for agro-hydrological studies in the semi-arid tropics of Tamil Nadu." Hydrology and Earth System Sciences. [https://hess.copernicus.org/articles/29/4847/2025/] — tested 8 products (TRMM, PERSIANN CDR, CMORPH, GPM-IMERG, MSWEP, NCEP2, ERA5-Land, MERRA2) against gauge data in Coimbatore/Madurai/Tiruchirappalli/Tuticorin (2003-2014). IMD was NOT one of the 8 tested — this paper cannot be used to compare IMD vs. ERA5-Land/MSWEP directly. Found ERA5-Land best overall in Tiruchirappalli (closest to our districts), MSWEP close second; MERRA2/NCEP2 worst everywhere. Also separately warns NASA POWER (MERRA2-based) tends to overestimate without bias correction — relevant caution for our own NASA POWER usage (humidity/ET), not something we've corrected for.
3. NASA POWER support (usage precedent, NOT an accuracy validation): a Tamil Nadu drought-monitoring study (SPEI index) used NASA POWER temperature+precipitation data 1991-2024, published in Plant Science Today (confirmed real, Scopus-indexed journal since 2018). Link: https://horizonepublishing.com/journals/index.php/PST/article/view/4653 — Important honest limitation: this paper never validates NASA POWER's accuracy against ground stations, and never uses humidity or ET at all (only temp+precip; ET was separately calculated via Thornthwaite method). Cannot be cited as proof NASA POWER's humidity/ET numbers specifically are accurate for our region — only as evidence NASA POWER is a trusted, real dataset used in real Tamil Nadu research.
4. General (non-region-specific) NASA POWER validation: a Portugal study found strong agreement (R²>0.82) for temp/radiation/humidity vs. real stations, weaker for wind speed; an Iran study found NASA POWER temperature performs worse at high-altitude/cold locations but better in low-altitude/warm regions and months — favorable for our low-altitude, warm Tamil Nadu setting, though not a direct regional test.
5. MODIS was considered as an alternative for humidity/ET (advisor asked "is NASA POWER standard, or should we use MODIS"). Finding: MODIS's MOD16A2 IS a real, standard, direct evapotranspiration product (fine to use). MODIS has NO direct humidity product — would require deriving RH from MOD07's water-vapor+temperature profile data, a multi-step calculation, with real cloud-cover gaps during monsoon season specifically (i.e. gaps exactly when it matters most for this project). Not pursued for humidity given this complexity; NASA POWER kept for humidity based on findings above.

**NEW PENDING DECISION — advisor wants multi-grid rainfall averaging instead of single-nearest-cell (raised verbally, NOT YET IMPLEMENTED, explicitly told to defer until spatial/QGIS-GEE visualization work is done first):**
- Context: while reviewing the Keelvalur spatial-matching example (well has 5 IMD grid cells within ~30km, only the single nearest at 2.6km is currently used), advisor asked whether the other 4 nearby cells also have rainfall data. Confirmed yes — all 18 grid cells project-wide have complete 3,287-day records, no gaps.
- Advisor's actual instruction (Toothless's recollection, paraphrased, not a direct quote — worth re-confirming exact wording later): "if the other grids also have data, we need to average across all of them" — i.e., change from single-nearest-cell matching to averaging rainfall across ALL nearby grid cells per well, for every well, not just Keelvalur.
- This is a SIGNIFICANT methodology change if implemented — moves from simple nearest-neighbor matching to a multi-point averaging approach (related to, but not the same as, formal Inverse Distance Weighting/IDW).
- OPEN QUESTIONS, unresolved, must be clarified before building anything:
  - How many nearby cells to include per well — all cells within a fixed distance (e.g. 30km, matching Keelvalur's case), or a fixed count (e.g. nearest 3, nearest 5) regardless of distance? Different wells will naturally have different numbers of "nearby" cells depending on local grid density near district edges.
  - Does this apply to rainfall only, or also temperature (which uses a separate, coarser IMD grid)?
  - Simple average across included cells, or distance-weighted (closer cells count more)?
- Advisor's own explicit sequencing instruction: finish the QGIS/Google Earth Engine spatial visualization work FIRST, then revisit this decision. Do not implement until she confirms these details.

**SUPERSEDED — kept for history only, no longer the plan (the MSWEP-switch discussion):**
IMD rainfall was briefly considered for replacement by MSWEP (v2.80, via gloh2o.org, accessed through a PhD scholar's shared Google Drive), with an extended 2011-2021 range, based on advisor's initial comment that "IMD datasets are wrong nowadays." After literature review (see above), advisor confirmed keeping IMD instead. MSWEP access/download work already done (Google Drive folder confirmed, 2010-2021 daily files partially downloaded) is not currently needed but the access remains available if reconsidered later.

**Spatial visualization work (advisor-requested, separate from Phase C build track) — STATUS:**
- DONE: Thanjavur district static map — 26 wells, 8 grid cells, real Esri satellite basemap (fetched via direct tile-stitching since contextily/rasterio is blocked by this machine's Windows Application Control policy), district boundary outline (from GADM, see below), connections, legend, scale bar, north arrow. 300 DPI. Saved: figures/thanjavur_well_grid_map.png + figures/thanjavur_well_grid_match.csv.
- DONE: Interactive Folium map — all 53 wells, all 18 grid cells, all 53 connections, all 3 district boundaries (toggleable), real satellite basemap, click-popups. Self-contained HTML (needs internet only to load satellite tiles). Saved: figures/interactive_well_grid_map.html. Verified: marker/line/boundary counts all confirmed correct.
- DONE: Full study-area static map (all 3 districts combined, one image). Saved: figures/full_study_area_map.png.
- DONE: District boundaries sourced from GADM v4.1 (official, free-for-academic-use administrative boundary database). Saved: data/tn_district_boundaries.gpkg. Note: GADM spells it "Nagappattinam" (double p) vs. this project's "Nagapattinam" — normalized to project spelling in the saved file, original GADM name kept alongside for traceability.
- NOT YET DONE / genuinely blocked: advisor specifically asked for this in QGIS or Google Earth Engine. Neither has been done yet:
  - QGIS: real desktop GIS software, point-and-click, NOT code — Claude Code can prepare data files (qgis_wells.csv, qgis_rain_grids.csv, qgis_connections.csv already generated) but cannot operate QGIS itself. Requires Toothless to manually build it by hand.
  - Google Earth Engine: investigated in detail. No GEE account exists yet on this machine (needs signup + Google Cloud project + a verification questionnaire with no guaranteed approval time, per Google's own current docs). Key finding: a LIVE/interactive GEE map's satellite tile layer expires after 24 hours (verified against GEE's real technical docs, not assumed) — NOT suitable as a permanent deliverable. Recommended approach instead: authenticate once, pull a real Sentinel-2 composite as a static image export, bake that into the EXISTING Folium/PNG pipeline (same pattern already used for Esri tiles) — genuinely GEE-sourced, but permanent, not live/expiring.
- Current plan: show advisor the working Folium map now as an interim/draft deliverable (being upfront it's Python/Folium, not QGIS or GEE), while separately pursuing the GEE static-composite approach and/or manual QGIS build to satisfy her actual tool request.


Turns the clean merged table into the exact numeric shape the model trains on. 5 sub-steps:

**C0 — Deduplication (locked, not yet coded).** 20 wells have 2 readings landing in the same quarter (e.g. Keelvalur, 1 day apart, both Q1 2019) — model predicts one value per well per quarter, so conflicts must be resolved. Rule: keep the most recent reading per quarter. Not averaged (arbitrary gap between duplicates), not interpolated (standing rule against manufacturing data).

**C1 — Feature vector definition.** x_i,t = [rainfall_180d_avg, tmax_90d_avg, tmin_90d_avg, humidity_90d_avg, et_90d_sum, depth_m], F = 6, where depth_m is the depth measured at that reading (the former prev_GWL / y_i,t wording is replaced); the target is next_depth = depth of the NEXT reading of the same well (decision 2026-10-14). Not every variable necessarily helps — sensitivity/importance pass across all candidates before locking the final list (answers advisor's "which of these matter most" question).

**C2 — Normalization (StandardScaler or MinMaxScaler).** Necessary because raw ranges differ wildly (rainfall 0-1205mm, temp 23-42°C, depth 0-20m) — without rescaling, a network effectively treats rainfall as "more important" purely because its numbers are bigger, not because it actually matters more.

**C3 — Sliding window construction, L=8 quarters (test 4 and 12 too).** Each training example = "8 consecutive quarters of features → predict quarter 9's groundwater level," per well. 8 is short for a Transformer (stated limitation) — testing 4/12 too lets the paper honestly report which length performs best rather than assuming. Real constraint: 9+ readings does not mean 9 consecutive quarters. On the extended data, strict consecutive windows = 157 from 23 wells; gap-tolerant windows = 840 from 42 wells (counts before the train/val/test split). Definition is OPEN, decide at C3 (see CURRENT STATE).

**C4 — Chronological 70/15/15 split (train=oldest, test=newest, NO random shuffling).** Random shuffling would let the model train on a 2020 reading and get tested on a 2015 one — testing on the "past" after learning the "future," which never happens in real forecasting and would fake the accuracy numbers. Chronological split keeps the test genuinely simulating predicting what hasn't happened yet.

### Phase D: Graph Construction
**What:** build a graph structure representing spatial relationships between the 53 wells — nodes = wells, edges = how strongly connected they are.
**Why it exists:** a GNN needs an explicit graph to operate on; without it, there's nothing to tell the model which wells might influence each other hydraulically.
- D1. Well coordinates (lat/long) — already available from CGWB.
- D2. Distance matrix: haversine distance between every well pair (real Earth-surface distance, not flat-plane approximation — matters at this geographic scale).
- D3. Adjacency: A_ij = exp(-d_ij^2 / sigma^2) — a Gaussian kernel. **Why this formula specifically:** it converts raw distance into a smooth similarity score — closer wells get an edge weight near 1, distant wells decay smoothly toward 0, rather than an arbitrary hard cutoff ("connected" vs "not") that would need justifying a threshold distance.
- D4. Normalize: A_hat = D^(-1/2) A D^(-1/2). **Why:** prevents wells with many close neighbors from numerically dominating the GCN's computation — standard stabilization for graph convolutions.
- D5. Output: graph (nodes=wells, edges=A_hat).

### Phase E: Temporal Encoding
**What:** prepares each well's own time sequence before it enters the Transformer.
**Why it's needed:** unlike an RNN/LSTM, a Transformer has no built-in sense of order — it sees a set of quarters, not a sequence, unless order is explicitly injected.
- E1. Per-well sequence X_i: [L x F] (L quarters, F features).
- E2. Sinusoidal positional encoding — injects "this is quarter 3 of 8" information into each timestep. Sinusoidal (not a learned embedding) chosen because it generalizes cleanly to window lengths not seen in training (relevant since L=4/8/12 are all being tested).
- E3. Linear embed → hidden dimension d_h = 64 — projects raw features into a higher-dimensional space, giving the model room to represent more complex patterns than the raw 5-6 input features alone allow.
- **Build order (2026-10-21):** the embedding is applied BEFORE the positional encoding, H0 = (X W_e + b_e) + PE, even though E2 is listed before E3. The PE table has dimension d_h = 64 while X has only 6 features, so PE can only be added after the embedding has projected X to 64 dimensions. The names E1, E2, E3 are kept.

### Phase F: GNN Encoder (spatial)
**What:** processes the well graph, producing a spatial embedding h_spatial for each well that reflects its neighbors' influence.
- F1. GCN Layer 1: H1 = sigma(A_hat X W0) — each well's new representation is a weighted combination of its neighbors' features (via A_hat), transformed by a learnable weight matrix W0, passed through a nonlinearity.
- F2. GCN Layer 2 → h_spatial. **Why exactly 2 layers, not more:** GNNs suffer from over-smoothing — stack too many layers and every well's representation converges to look the same, losing the local distinctiveness the whole architecture exists to capture. 2 layers is the standard safe depth.

### Phase G: Transformer Encoder (temporal)
**What:** processes each well's own history, learning which past quarters matter most for predicting the next one.
**Why attention over an LSTM:** attention lets the model weigh ALL past quarters directly and simultaneously — a monsoon from 6 quarters back can still matter, without having to survive being passed step-by-step through a recurrent chain.
- G1. Multi-head self-attention (4 heads): softmax(QK^T / sqrt(d_k)) V. **Why multiple heads:** each head can specialize — one might track rainfall trends, another seasonal humidity patterns — rather than forcing one attention mechanism to capture everything at once.
- G2. Add & Norm → Feed-forward → Add & Norm (2 layers, dropout 0.2) → h_temporal. Residual connections (Add) prevent vanishing gradients; LayerNorm stabilizes training; dropout fights overfitting on a relatively small dataset.

### Phase H: Late Fusion (always name it this)
**What:** u_i,t = W_f [h_spatial ; h_temporal] + b_f — concatenate both encoder outputs, pass through a joint linear/MLP decoder.
**Why "late" fusion specifically:** keeps the spatial and temporal branches fully independent until the very end. This isn't just a design preference — it's what makes the ablation studies (Phase L) clean and honest: you can remove the GNN branch or the Transformer branch entirely and the other still works, which wouldn't be possible if they were mixed earlier (early/alternating fusion).

### Phase I: Prediction Head + Loss
Target: groundwater level, NEXT QUARTER (T+1).
- I1. y_hat_i,T+1 = W_p u_i,T + b_p — a final linear layer mapping the fused representation to one predicted number.
- I2. Loss = MSE(y_hat, y) + lambda * L2 regularization. MSE is the standard choice for continuous regression, penalizing larger errors more. L2 regularization guards against overfitting — important given only 53 wells, a small dataset by deep learning standards.

### Phase J: Training Setup
hidden dim=64, GNN layers=2, Transformer layers=2, attention heads=4, dropout=0.2, window L=8, batch size=16–32, Adam optimizer, learning rate=1e-4, epochs=100–200 with early stopping.
**Why halved vs. Hua et al.'s A100 setup:** explicit RTX 3050 (6GB VRAM) compute-budget decision — state this openly in the paper as a constraint, not hide it. Early stopping specifically guards against overfitting on the small well count.

### Phase K: Baselines
ARIMA/SARIMA, persistence (next quarter = last reading; added 2026-10-13 because prev_depth alone has r about 0.78), plain per-well LSTM, plain per-well CNN, GNN+LSTM (no attention), Transformer-only (optional, no graph).
**Why baselines matter, specifically for this project:** this is the direct, honest answer to "why GNN+Transformer instead of something simpler" — baselines prove (or disprove) that the added architectural complexity actually earns better accuracy, rather than being complexity for its own sake. LSTM/CNN-per-well specifically matches what every current India-focused paper in the literature review already does — showing your model against "what everyone else does" is the whole point of the comparison.
Modern STGNNs (Graph WaveNet, DCRNN) explicitly out of scope — future work only.

### Phase L: Ablations
Full model (GNN+Transformer+Fusion) / without GNN (Transformer only) / without Transformer (GNN+plain LSTM instead).
**Why distinct from baselines:** baselines are entirely different architectures; ablations are controlled variants of your OWN model with one piece removed. This isolates exactly which component (spatial or temporal) is responsible for how much of the accuracy — a much more precise claim than "our model beats LSTM."

### Phase M: Robustness
Missing data (10/30/50% readings dropped), unseen wells (never seen during training), recent unseen years.
**Why this phase exists:** real deployment is messy — sensors fail, new wells get added, and the model eventually has to predict years beyond its training data. These tests answer a reviewer's likely question directly: does this actually hold up outside a clean lab setting, not just on the exact data it was trained on.

### Phase N: Evaluation Metrics
MAE (report in meters, e.g. "predicts within 1.2m"), RMSE, R². Report per-well and averaged. R² 0.80–0.92 is internal target only, never stated as a target in the paper — report the actual honest result.
**Why MAE in meters, not normalized units:** interpretability — "predicts within 1.2 meters" means something to a hydrologist or policymaker; a normalized error score doesn't.
**Why the extra physical plausibility + spatial coherence checks, beyond the statistical metrics:** a model can post a good R² while still making physically nonsensical predictions (e.g. water rising during a drought) — statistical accuracy alone wouldn't catch that; these checks specifically test whether the model learned real hydrology or just curve-fit the numbers.

### Phase O: Output
Predicted GWL → per-well forecast → district average (Thanjavur / Thiruvarur / Nagapattinam separately) → risk classification (Safe/Semi-critical/Critical/Over-exploited, matching CGWB's own categories) → visualization.
**Why this phase matters beyond just "presenting results":** it connects the model's raw numeric output back to CGWB's real, official decision-making categories — the same system already confirming Thanjavur as over-exploited and Nagapattinam as excluded entirely due to salinity. This makes the project a usable early-warning tool, not just an academic accuracy exercise.

### Phase P: Explainability
Optional, future work only — do not build unless everything else is done with time remaining. Not essential to prove the core hypothesis (that spatial+temporal fusion beats temporal-only), so it doesn't block publication if time runs out.

## Paper Writing Plan
Target: HYDRO INTERNATIONAL 2026, IIT Indore, 17-19 Dec 2026. **Format/page limit not yet confirmed** - check hydro.iiti.ac.in directly or ask advisor before finalizing length. Do NOT use deadlines from hydropower-dams.com's "HYDRO 2026" - that is a different, unrelated UK conference.

Figure 1 (full architecture diagram - all locked equations, verified against CLAUDE.md, accurate) - DONE, built via eraser.io.

### Final Print Structure (how the finished paper reads, top to bottom)
1. Title
2. Abstract
3. Introduction
4. Related Work
5. Study Area and Data
6. Methodology
7. Experimental Setup
8. Results and Discussion
9. Conclusion and Future Work
10. References

### Drafting Order (the order we actually WRITE sections in — deliberately different from the print order above)
**Why different:** Introduction needs to state the literature gap, which only exists once Related Work is actually drafted — so Introduction can't be written first even though it's read first. We draft in the order each section becomes possible, then assemble into the Final Print Structure above at the end. Nothing is lost, this is normal academic writing sequencing, not a mistake.

1. Study Area & Data — ready now, 100% real numbers available. **STATUS: REWRITTEN 2026-10-13 for the extended 2013-2025 data and inserted into master_draft.md (Study Area, Data Sources, Data Integration, Data Limitations; Tables 1-4; refs [15]-[20]). Refs [17], [19], [20] have placeholders to confirm. Needs user review.**
2. Related Work — ready now, 13 papers read, gap table exists. NEXT.
3. Introduction — needs #1+#2 finished first (uses the confirmed gap from Related Work).
4. Methodology — see phase mapping below.
5. Experimental Setup — see phase mapping below.
6. Results & Discussion — WAIT for real training results (Phase J+).
7. Conclusion & Future Work — partial now (future work half), finish after results.
8. Abstract — always last, needs every other section finished (summarizes problem + method + actual result).

### Phase-to-Section Mapping (which build phases feed which paper section)
| Paper section | Fed by build phase(s) | Ready to draft once... |
|---|---|---|
| Study Area & Data | Phase A (acquisition), Phase B (fusion) | A+B done — TRUE now |
| Related Work | (literature review step, pre-Phase-A) | 13 papers read — TRUE now |
| Introduction | Related Work's gap + Study Area's motivation | #1+#2 drafted — TRUE now |
| Methodology | Phase C (preprocessing) through Phase I (prediction head + loss) | All of C–I locked in this file — TRUE now (equations already written for D–I; C's formulas get added as each sub-step is built) |
| Experimental Setup | Phase J (training setup), plus the *design* of Phase K/L/M/N (baselines, ablations, robustness, metrics — describing what will be run, not results yet) | J locked, K/L/M/N design locked — TRUE now |
| Results & Discussion | Actual output of Phase J (training) through Phase O (output/risk classification) | Real numbers exist — FALSE, blocked until training runs |
| Conclusion & Future Work | Everything above + Phase P (explainability, if built) | Results exist for the "conclusion" half; "future work" half can be written now | 
| Abstract | Every section above | Everything else finished |

Requirements: real equations (not simplified), LaTeX-quality math notation, real figures, humanized prose (user's own voice preserved, not generic AI tone), Scopus-standard rigor matching Hua et al. 2025. Every number/claim must trace to a "Real Confirmed" line in this file - never invented or approximated.

Workflow: Claude Cowork writes each section's first draft directly, from a prompt built on CLAUDE.md's "Real Confirmed" numbers/decisions plus the 13 read papers. Toothless + Claude (chat) review the draft for factual accuracy against CLAUDE.md and for tone/clarity. Any mistakes found are written up as a correction prompt and sent back to Cowork. Repeat until clean, then Toothless does a final read.

**Master draft file:** keep ONE running document with all 10 Final Print Structure headings present from the start (Title, Abstract [placeholder], Introduction [placeholder], ... References [placeholder]). As each section finishes drafting, it gets dropped into its correct slot in this same file — so the document is always readable top-to-bottom in final order, even while we're still writing out of order behind the scenes.

## Known Limitations (state honestly in the paper)
1. 8-quarter window is short for a Transformer — not fixed via interpolation. Window sizes 4/8/12 tested and compared honestly instead.
2. Per-well reading density is uneven (extended data: median 26 readings, range 1-43 over about 12 years; 11 wells have fewer than 9 readings).
3. 20 well-quarter duplicates resolved by keeping the most recent reading — not interpolation.
4. Groundwater readings end 2024-11-30 (portal file stops 2025-01-30), so forecasting means one quarter ahead on held-out past years, not future years.
5. 19 of 53 wells have no readings after Jan 2021.
6. Six-hourly telemetry data was investigated and deliberately not used (see CURRENT STATE).
7. If the gap-tolerant window is chosen, a window is the 8 most recent quarterly readings with the time gap as an input, not strictly 8 consecutive quarters.
8. Our original 838 rows are dated 1 day earlier than the portal's dates; new rows use portal dates.

## Folder Structure
```
data/           -> ONLY the 5 final 2013-2025 files (see CURRENT STATE). Old files, scripts, reports, raw downloads are in archive/data_build/.
preprocessing/  -> c0_dedup.py, c1_features.py, c4_split.py, c2_normalise.py, c3_windows.py (run in this order: C0, C1, C4, C2, C3) with outputs c0_dedup.csv, c0_removed_duplicates.csv, c1_features.csv, c4_split_dates.json, c2_scaler.json, c2_normalised.csv, c3_windows_gap.npz, c3_windows_strict.npz and a cN_report.md for each step
graph/          -> distance_matrix.py, graph_builder.py
models/         -> gnn.py, transformer.py, fusion.py, model.py
training/       -> train.py, baselines.py, ablations.py, robustness.py, evaluate.py
output/         -> forecast.py, risk_classification.py, visualize.py
```

---

## PROGRESS
- [x] Phase A — All 4 data sources confirmed downloaded and cleaned.
- [x] Phase B — Merge done and verified (real numbers above).
- [x] Dataset extension (groundwater to 2024-11, weather to 2025-12) — DONE 2026-10-13, regression check passed.
- [x] Data folder cleaned (5 files in data/, rest in archive/data_build/) — DONE 2026-10-13.
- [ ] Phase C — Preprocessing (dedup + feature engineering + sensitivity pass)
  - [x] C0 dedup: 1,211 readings after removing same-quarter duplicates (keep most recent).
  - [x] C1 features and target: 1,158 samples with a target (1,211 minus 53 last readings); features depth_m + 5 weather, target next_depth.
  - [x] C4 split (done before C2): cutoffs 2022-09-30 (train/val) and 2023-12-31 (val/test), by target date. GAP windows train/val/test = 581/147/112 (69.2/17.5/13.3%); STRICT (8 consecutive quarters + target in the next quarter) = 80/26/51.
  - [x] C2 z-score: fitted on 952 readings dated on or before 2022-09-30; depth_m and next_depth share statistics; scaler in c2_scaler.json.
  - [x] C3 windows (L=8): GAP 840 windows / 42 wells (reproduces the hand count); STRICT 157 windows / 23 wells (8 consecutive quarters AND target gap 1, decision 2026-10-18; reproduces the hand count; 180 / 25 without the target-gap rule). Persistence MAE GAP val 0.888 m, test 1.347 m.
- [x] Phase D — Graph construction (done 2026-10-20)
  - [x] D2 distance matrix: haversine, R = 6371.0088 km, 53 x 53; off-diagonal min 0 / median 49.137 / max 125.312 km.
  - [x] D3 + D4 adjacency A and A_hat with sigma = 24.7975 km (std of off-diagonal distances; chosen by user 2026-10-20 over 10 km, 25 km and median 49.137 km). All checks passed. 2 zero-distance pairs kept as separate nodes (Icchankottai/Icchankottai1, Vedaranyam/Vedaranyam1).
- [x] Phase E — Temporal encoding (built 2026-10-21, explained E1-E3, paper section written 2026-10-25). E3 viva question (shape and content of the final 6-number row after E3) still open.
- [ ] Phase F — GNN encoder (built 2026-10-25, paper section written 2026-10-25; F1-F3 teaching in progress, F1 explained, F1 viva pending; tick after F3 is understood)
- [ ] Phase G — Transformer encoder (built 2026-10-29, paper section written 2026-10-30; teaching in progress, user asked for SHORT/MEDIUM explanations paragraph by paragraph from the paper section; tick after the user understands)
- [ ] Phase H — Late fusion
- [ ] Phase I — Prediction head + loss
- [ ] Phase J — Training setup
- [ ] Phase K — Baselines
- [ ] Phase L — Ablations
- [ ] Phase M — Robustness checks
- [ ] Phase N — Evaluation metrics
- [ ] Phase O — Output
- [ ] Phase P — Explainability (optional)

---

## CURRENT STATE (2026-10-13) - read this first

### Working data files (data/ contains ONLY these 5)
- `cgwb_thanjavur_thiruvarur_nagapattinam_2013_2025.csv`: 1,232 rows, 53 wells, 2013-01-01 to 2024-11-30. Schema: date, state_name, district_name, station_name, latitude, longitude, basin, sub_basin, source, currentlevel, level_diff.
- `imd_rain_thanjavur_thiruvarur_nagapattinam_2013_2025.csv`: 85,464 rows, 18 grid cells (0.25 deg), 2013-01-01 to 2025-12-31.
- `imd_temp_thanjavur_thiruvarur_nagapattinam_2013_2025.csv`: 42,732 rows, 9 grid cells (1 deg), same dates.
- `nasa_power_humidity_et_2013_2025.csv`: 251,644 rows, 53 wells, same dates.
- `merged_final_2013_2025.csv`: 1,232 rows, 0 missing values. Columns: well_id, date, rainfall_180d_avg_mm, tmax_90d_avg_c, tmin_90d_avg_c, humidity_90d_avg_pct, et_90d_sum_mm, depth_to_water_level_m. ALL later phases read ONLY this file.
- Everything else (old 2013-2021 files, scripts, audit and extension reports, raw downloads, the 469 MB telemetry file, backups, GADM boundaries) is in `archive/data_build/`. Nothing was deleted. Scripts there read files from their own folder, so to re-run one, copy it back with its inputs. `output/map_*.py` import `merge.py` from `data/` and will fail until `merge.py` is copied back.

### Dataset extension (DONE 2026-10-13)
- Source: National Water Data Portal (nwdp.nwic.gov.in), CGWB Tamil Nadu quarterly manual groundwater level files (1991-2020 and 2021-2025).
- 425 rows matched our 53 well names EXACTLY after 2021-01-09. Dropped 3 empty values. Excluded 28 Jan-2021 boundary rows (same measurements as our existing 2021-01-09 rows, dated one day later on the portal; values identical). Added 394 readings, dated 2021-01-10 to 2024-11-30, for 34 wells.
- 19 wells have no new readings: 17 not present in the portal file under the same name, Tiruvarur (only an empty value), Vedaranyam1 (only a boundary duplicate). No fuzzy matching was done.
- Date convention: our original 838 rows are dated 1 day earlier than the portal's (789 of 790 matched values identical after a +1 day shift). New rows keep portal dates. One new reading (Vallam 2023-01-01) would change quarter under a 1-day shift. `data/cgwb.py` never existed and no download URL was recorded, so cite the portal as the source.
- Weather: IMD rain/temp and NASA POWER extended to 2025-12-31, no gaps; 2021 re-derived bit-identical to the old files. `imdlib` cannot be imported (scipy blocked by Windows Application Control), so a numpy-only parser was used with the same endpoints, boxes and fill-value rules.
- Regression check PASSED: all 838 original merged rows reproduced exactly (0 differences).
- Verified counts (extended file): wells per district Thanjavur 26, Nagapattinam 16, Thiruvarur 11; readings per district 552 / 471 / 209; readings per well min 1, median 26, max 43; 42 wells with 9+ readings, 33 with 17+. Depth range -0.9 to 21.32 m, median 3.66, mean 4.67.
- Merged feature ranges: rainfall_180d_avg_mm 0.0 to 10.33; tmax_90d 30.01 to 37.68; tmin_90d 21.91 to 27.0; humidity_90d 52.08 to 88.02; et_90d_sum 0.02 to 159.38.
- Incomplete windows: 51 rows (90-day windows) and 88 rows (180-day rainfall window), all at the very start of the data (Jan 2013); none in the new rows.
- Same-quarter duplicates: 21 in the extended file (1,232 readings minus 1,211 occupied well-quarter slots). C0 is NOT applied yet.
- Flagged, not removed: 3 negative depths, all Allankottai (-0.74 on 2021-01-10, -0.77 on 2023-01-20, -0.9 on 2024-11-30). Decide at C0/C1.

### Telemetry (investigated, NOT used)
- TN SW&GW six-hourly telemetry 2021-2025: 468.7 MB, 849 stations, 69 in our 4 districts (incl. Mayiladuthurai). Not our wells: 2 of our 53 wells have a station within 1 km, 34 within 5 km, 48 within 10 km. Sign convention undocumented (97.9% of values negative); at the one comparable pair (Puduchataram, 3 dates) the sign is opposite and the magnitude differs. Sentinel values (-999.999) and outliers present; 36 of 69 target stations start only in 2024. Verdict: future work only. File archived.

### Window definition (OPEN, decide at C3)
- Strict consecutive 8 quarters: 157 windows from 23 wells (80 from 16 wells before extension).
- Gap-tolerant (8 most recent occupied quarterly slots, gaps allowed, time gap in quarters added as an input feature, as in Hua's time-interval-aware attention): 840 windows from 42 wells (447 before extension).
- Counts are before the 70/15/15 split. No well has 17 consecutive quarters, so L=16 is only possible gap-tolerant.
- Recommendation: gap-tolerant with the time-gap feature; run strict as a robustness check. The paper must say "8 most recent quarterly readings".

### Phase D result (2026-10-20)
- Node order = ascending well_id string order (same order c3_windows.py iterates the wells), stored in `graph/well_order.json`. Later code must LOAD this file and never recompute the order.
- Files: `graph/distance_matrix.npy`, `graph/adjacency.npy`, `graph/a_hat.npy`, `graph/graph_meta.json`, `graph/d_report.md` (plus `graph/well_order.json`, and the scripts `graph/distance_matrix.py`, `graph/graph_builder.py`).
- Cross-district share of off-diagonal A weight: 35.7%. Neighbours per well: mean 6.49 at A > 0.5 and 17.66 at A > 0.1.
- Most isolated wells (smallest row sum of A): Peravurani, Vedaranyam, Vedaranyam1, Puttur, Thagattur.

### Phase E build result (2026-10-21)
- E1: X (840, 8, 6), 581 / 147 / 112 windows, 42 wells, 0 NaN in X, 42 NaN in G (all in G[:,0]); E2: PE table (8, 64) saved as `encoding/pe_table.npy`, all checks passed; E3: torch 2.6.0+cu124 installed into `.venv` on user request (2026-10-21), import works and CUDA is available; nn.Linear(6, 64) + PE gives output (4, 8, 64), 448 parameters, max difference to the numpy hand calculation 2.4e-07; all checks passed (ALL CHECKS PASSED). Files: `encoding/e_temporal.py`, `encoding/run_phase_e.py`, `encoding/e_report.md`, `run_phase_e.bat`.

### Phase F build result (2026-10-25)
- Node-feature design (final, decided 2026-10-25): for window w with target well i and time t = last_input_date[w], every one of the 53 wells (order from `graph/well_order.json`) gets 8 values: the 6 normalised features (same order as X) of its latest reading with date <= t (last observation carried forward, LOCF); staleness = (t - date of that reading) in days / 91.25 (quarters), z-scored with the TRAIN-windows-only mean 4.6595 and std 8.3069 (30,421 observed node values in 581 train windows, `gnn/f_staleness_scaler.json`); has_reading = 1 if any reading with date <= t exists, else 0. Wells with no reading <= t get all zeros. Only readings dated <= t are used (no leakage).
- Encoder: `GCNEncoder`, in 8, hidden 64, out 64, 2 layers, Kipf and Welling propagation with NO extra identity (A_hat already has a self weight): H1 = ReLU(A_hat (H0 W1) + b1), dropout 0.2, H2 = ReLU(A_hat (H1 W2) + b2); bias added after the A_hat product; A_hat loaded from `graph/a_hat.npy`, never rebuilt. Input (B, 53, 8) -> Z (B, 53, 64); `target_embedding(Z, node_idx)` returns the (B, 64) row of the target well for the late fusion in Phase H. Parameters 4,736. Nothing trained.
- Key numbers: NF (840, 53, 8), no NaN; max (date used - t) = 0 days; target well's own row equals the last window position (max difference 1.7e-07, float32); observed nodes per window min / median / max: train 52 / 52 / 53, val 52 / 53 / 53, test 53 / 53 / 53; staleness of carried readings up to 37.7 (train), 42.7 (val), 46.7 (test) quarters; the 11 wells with no window are observed as neighbours in all 840 windows except Vedaranyam1 (467); numpy hand check max difference 2.5e-07; CPU and CUDA (RTX 3050) agree within 1.8e-07; Keelvalur test: +1.0 on a feature of the nearest neighbour (Nagapattinam1, A_hat 0.0984) changes z_target by 0.042, on the farthest well (Puduchataram, A_hat 8.2e-07) by 1.0e-04; two-layer reach A2 = A_hat @ A_hat: self weight min 0.0449 (Valangiman), median 0.0606, max 0.2365 (Vedaranyam1), all 53 wells have more than 50% of their A2 row weight outside themselves.
- Files: `gnn/f_node_features.py`, `gnn/f_node_features.npz`, `gnn/f_staleness_scaler.json`, `gnn/f_gcn.py`, `gnn/run_phase_f.py`, `gnn/f_report.md`, `gnn/f_results.json`, `run_phase_f.bat`. One command: `.\run_phase_f.bat`; ends with ALL CHECKS PASSED. The Phase F progress box is not ticked (the user ticks it after learning).

### Phase G build result (2026-10-29)
- Design (final, decided 2026-10-29): embedding `nn.Linear(6, 64)` plus the fixed sinusoidal PE (non-trainable buffer loaded from `encoding/pe_table.npy`, H0 = X W_e + b_e + PE, as in Phase E); `nn.TransformerEncoder` of 2 x `TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.2, activation="gelu", batch_first=True, norm_first=True)` plus a final `LayerNorm(64)`, `enable_nested_tensor=False`; no attention mask. Output Hn (B, 8, 64); h_target = Hn[:, -1, :] (latest reading), (B, 64), for the late fusion in Phase H. The window length L is a constructor argument (default 8; L = 4 and 12 build and run; the PE for L != 8 uses the same formula, and PE(8) equals the Phase E table exactly). Nothing trained.
- Parameters: embedding 448; per layer 33,472 = 16,640 (attention) + 16,576 (feed-forward) + 256 (two LayerNorms); two layers 66,944; final LayerNorm 128; encoder 67,072; with the embedding 67,520. The module's counts equal the formula counts.
- Key numbers: numpy hand check of the whole encoder (embedding + PE, pre-norm attention with softmax(QK^T/sqrt(d_k))V for 4 heads, residuals, GELU feed-forward, final LayerNorm) max difference 6.2e-07; attention rows sum to 1 within 1.8e-07 and all weights >= 0 (untrained weights, no meaning yet); permutation test: without PE, shuffling the first 7 readings changes h_target by 4.8e-07 (none), with PE by 0.123; dropout active in train mode, none in eval; gradients finite and non-zero for all 28 parameter tensors; CPU and CUDA (RTX 3050) agree within 7.2e-07; h_target size 64 = GCN z_target size 64 (same 840 windows); forward of 128 windows on the GPU about 2 ms, peak 11 MiB (timing varies).
- Fast-path finding: with PyTorch's default fused eval-mode kernel (eval + no_grad) the CUDA result differed from the CPU and float64 results by 2.7e-04; the module now switches that fused path off while it runs (`no_fastpath()` in `g_transformer.py`, setting restored afterwards), so eval and train use the same arithmetic and CPU, GPU and the numpy hand calculation agree to about 1e-6. The architecture is unchanged.
- Note on the gradient check: loss = h_target.sum() gives about zero gradient inside the layers (the sum of a LayerNorm output with weight 1 and bias 0 is always 0); a fixed random projection of h_target is used as the informative test.
- Files: `transformer/g_transformer.py`, `transformer/run_phase_g.py`, `transformer/g_report.md`, `transformer/g_results.json`, `run_phase_g.bat`. One command: `.\run_phase_g.bat`; ends with ALL CHECKS PASSED. The Phase G progress box is not ticked (the user ticks it after learning).

### Other open items
- Risk-class mapping: CGWB categories are defined by percentage of recharge extracted, not by depth. Phase O needs an explicit, stated depth-to-category rule, and Nagapattinam (excluded as saline) must be handled separately.
- `import torch` in `.venv` now works (torch 2.6.0+cu124, CUDA available, 2026-10-21); no fallback to Colab or Kaggle needed so far.
- Optional: careful name/coordinate match for the 17 wells with no new readings (not done).
- Multi-grid rainfall averaging: still deferred, per the advisor's sequencing.
- Sigma sensitivity test (10 / 25 / 49 km) to run in Phase L and report in the paper.
- Stale weather features at LOCF readings (a well's carried reading can be many quarters old; staleness and has_reading flag it but the weather values are still old) is a limitation to state in the paper.
- Phase L ablation of GCN node features: A (static coordinates only) vs B (LOCF + staleness + has_reading, the main design).
- GCN node features option C (fresh weather at time t for every well) is a possible extension, not built.
- Phase L ablation: Transformer output at the last position (h_target = Hn[:, -1, :], main design) vs mean pooling over the 8 positions.
- The positional encoding counts readings, not time (the gap array G is unused by the Transformer); a time-interval-aware positional encoding is a possible extension.
- The Transformer has no attention mask because every position in a window is a reading on or before the forecast time.
- `enable_nested_tensor=False` is set because the nested-tensor fast path does not work with norm_first=True (and there is no padding to skip).
- Gap information (quarters between readings) is not used by the Phase E encoding; decide later whether to use a time-interval-aware positional encoding.
- GCN node features X for wells with no reading at a given time (11 wells with fewer than 8 readings, 19 wells with no reading after Jan 2021) DECIDED 2026-10-25: last observation carried forward + staleness + has_reading (see Phase F build result).
- The advisor was not consulted about the date-range extension (user decision, 2026-10-13).
- HYDRO INTERNATIONAL 2026 abstract (500 words) was drafted and reviewed on 2026-09-18; final wording says "more than six times larger" well network. The claim is about network size, not dataset size: Hua et al. have far denser readings per well.

### Time plan (user stated 15 days remaining on 2026-10-13)
- Build: Block 1 preprocessing and graph 1 day; Block 2 model and first training 2; Block 3 baselines (with persistence) and ablations 2; Block 4 robustness, metrics, output 2; window-size tests (4/8/12/16) 1. Build total 8 days.
- Writing: 2 days. Buffer: 1 day. Total planned: 11 of 15 days.

### Paper figures (2026-10-13)
- `figures/fig1_study_area.png/.pdf` (Figure 1, study area by district) and `figures/fig2_data_integration.png/.pdf` (Figure 2, rain and temperature nearest-cell matching) are made by `output/make_paper_figures_1_2.py`, reading only the files in data/ plus the boundary file in archive/data_build/. The architecture diagram becomes Figure 3.
- Old maps and scripts moved to `archive/old_figures/`. `figures/well_grid_matching_all53.xlsx` kept as a possible supplementary table.
- `figures/fig3_architecture.png/.pdf` (Figure 3, model only). The workflow overview figure (old Figure 4) was dropped on 2026-10-14 and its files moved to `archive/old_figures/`. The old Eraser diagram is in `archive/old_figures/architecture_diagram_old.png` (outdated: 838 readings, 90-day rain sum). Open: what the GCN node features X are (decide in Phase 3).
- UPDATE 2026-10-14: Figure 3 was redrawn (Hua-style: real 53-well graph, token strips, GCN layers, attention map marked illustrative, fusion, head, real Lower Anaicut example). Script: `output/make_fig3_architecture.py` (run from the project root). Figure 3 covers Phases D to I only. Phases A-B live in Study Area and Data, C in Methodology part 1, J-N in Experimental Setup and Results, O in Results.

## PAPER PDF RULE AND PHASE C PAPER STATUS (2026-10-19)
- The paper the user reads is the PDF `Groundwater_GNN_Transformer_Paper.pdf` in the project root. It is built from `master_draft.md` (the source). After every phase: write the paper section into master_draft.md, rebuild the PDF, and re-write the Project copy `claude/paper_master_draft.md` so the copies do not drift.
- Phase C paper section WRITTEN 2026-10-19: Methodology > Data Preparation (duplicates, inputs and target, chronological split, normalisation, sliding windows, leakage checks). Equations (1)-(9), Tables 6-8, Figure 3. Backup of the older draft: archive/master_draft_before_phaseC.md.
- Figure numbering in the paper: Figure 1 study area, Figure 2 data integration, Figure 3 = figures/figC_split_and_windows.png (script output/make_figC_split_windows.py, run from the project root), Figure 4 = figures/fig4_positional_encoding.png (Phase E, script output/make_fig_pe.py), Figure 5 = the architecture drawing (use figures/architecture_detailed.png after Phase I; old fig3_architecture.png is outdated). UPDATE 2026-10-25: the architecture figure moved from Figure 4 to Figure 5.
- Final training dataset: preprocessing/c3_windows_gap.npz (840 GAP windows, 581/147/112). Readable copy: preprocessing/final_dataset/final_dataset_gap.xlsx (sheet Training_data). STRICT (157 windows) is a sensitivity check only.
- Open before the model phases: NaN in G[:,0] for 42 GAP windows; GCN node features X; negative Allankottai depths check; risk-class rule; the 11 wells with fewer than 8 readings.

## PAPER STATUS UPDATE: PHASE D GRAPH SECTION (2026-10-21)
- Methodology > Graph Construction written in master_draft.md (Eqs. 10 and 11, Tables 9 and 10, reference [21] Kipf and Welling, to confirm against original). PDF rebuilt (15 pages).
- Eq. (1) in Study Area says R = 6371 km; graph distances use R = 6371.0088 km (stated in Graph Construction). Difference negligible.
- Backup: archive/master_draft_before_phaseD.md.
- One-command run for Phase D: .\run_phase_d.bat (graph/run_phase_d.py). Same pattern planned for each later phase, plus a master run_all later.
- Open: Phase D viva D5 still unanswered by user; graph text says all 53 wells are nodes including 11 with no window; decided in Phase F (LOCF, 10 of the 11 are observed in all 840 windows, Vedaranyam1 in 467).

## PROGRESS NOTE (2026-10-24): DETAILED ARCHITECTURE DIAGRAM + STATUS TILL E2
- Detailed architecture figure built (Hua et al. 2025 style): columns 1 Data layer, 2 Preparation, 3 Two-branch encoder (spatial GCN / temporal Transformer), 4 Prediction, plus bottom row 5 with J Training setup, K Baselines, L Ablations, M Robustness, N Metrics, O Output.
- Files in figures/: architecture_detailed.png and .pdf (clean, for paper Figure 5 after Phase I), architecture_detailed_progress.png (badges: green = built A-E, grey = not built F-O), make_architecture_diagram.py (regenerate with: python make_architecture_diagram.py <project_root> <out_prefix> [progress]; reads real files: well_order.json, a_hat.npy, pe_table.npy, c3_windows_gap.npz, c2_scaler.json). Mini-visuals use real data (53-well map, Keelvalur window, PE table); the attention matrix is illustrative only (labelled).
- Update badges (F, G, H, I, J..O) to green when each phase is built, then regenerate. Old fig3_architecture.png still has outdated label "previous depth"; do not use in paper.
- Open item for Phase O: risk-class rule (depth to CGWB category) is undecided; figure shows only the class names.
- Learning status: Phase E overview, E1, E2 explained. E2 viva pending (why PE row 0 = [0,1,0,1,...]). E3 (embedding 6 to 64, 448 params, H0 = XW_e + b_e + PE) being explained 2026-10-24. SUPERSEDED 2026-10-25: E2 and E3 understood; Phase E ticked.

## PROGRESS NOTE (2026-10-25): PHASE E PAPER SECTION WRITTEN
- Paper: Methodology > Temporal Encoding added (subsections: why the order has to be supplied, embedding, positional encoding, model input, checks, limitation). Eqs. (12) attention [22], (13) embedding e = xW_e + b_e, (14) sinusoidal PE, (15) H0 = (W W_e + 1 b_e^T) + PE. Table 11 (quantities), Figure 4 (PE table + distance between position vectors). New reference [22] Vaswani et al. 2017. Next equation number is (16); next table is 12; next figure is 5 (architecture).
- Numbers used (all checked against files): 448 parameters; PE range -0.990 to 1.000; row distances 1.472 (adjacent, same for all adjacent pairs), 2.719, 3.581, 4.180 (positions 0 and 7); hand-vs-PyTorch max difference 2.4e-07; Keelvalur window 39 depths 4.01, 2.91, 3.96, 4.51, 2.40, 1.68, 4.23, 4.51 m, target 5.10 m, gaps NaN,1,1,1,2,3,2,1 (train split).
- Limitation written in the paper: PE counts readings, not time; gap array G stored but unused; Hua et al. use a time-interval bias; time-interval-aware PE listed as possible extension.
- PDF footer text now lists: Introduction, Related Work, Study Area and Data, Data Preparation, Graph Construction, Temporal Encoding. PDF is 20 pages.
- Open vivas: C2, C3, C4, D5, E overview, E1, E3. Next: finish teaching Phase F (F1 explained; F2 GCN layers and F3 target row remain).

## PROGRESS NOTE (2026-10-25, evening): PHASE F PAPER SECTION WRITTEN
- Paper: Methodology > Spatial Encoding added (node features at the forecast time, graph convolution, representation of the target well, reach of the two layers, checks, limitations). Eqs. (16) node vector and staleness, (17) two GCN layers, (18) target row. Tables 12 (node features by split) and 13 (checks). Next equation number (19), next table 14, next figure 5 (architecture).
- Numbers in the section (all from gnn/f_report.md): 44,520 window-node pairs, 373 unobserved (0.8%); staleness mean 4.660, std 8.307 quarters over 30,421 observed train values; max age 37.65 / 42.66 / 46.67 quarters (train / val / test); 4,736 parameters; hand-vs-module 2.5e-07; CPU vs GPU 1.8e-07; influence test Keelvalur window 39: Nagapattinam1 0.042 vs Puduchataram 1.1e-04; A2 self weight min 0.0449 (Valangiman), median 0.0606, max 0.2365 (Vedaranyam1), all 53 wells have more than 50 percent of the row weight outside themselves.
- Claims softened on purpose: third GCN layer not tested; Thillaivilagam is not among the five least connected wells.
- Paper limitations written: stale carried readings and their stale weather; ablation planned in Phase L (static coordinates vs LOCF); fresh-weather node features not built.
- PDF footer lists: Introduction, Related Work, Study Area and Data, Data Preparation, Graph Construction, Temporal Encoding, Spatial Encoding.
- Teaching status: F1 explained, check question pending (how Nagapattinam1 staleness 1.195 arises). The user said they have one important thing to tell after the Phase F paper section.

## PROGRESS NOTE (2026-10-25, night): LATEX VERSION FOR OVERLEAF
- latex/main.tex is generated from master_draft.md by latex/make_latex.py (pandoc based; run from the project root after copying the 4 figure PNGs into latex/figures). Do NOT edit main.tex by hand; edit master_draft.md and regenerate. latex/overleaf_paper.zip = main.tex + figures, upload with Overleaf > New Project > Upload Project (compiler pdfLaTeX).
- Generic article class (11pt, A4, single column), numbered equations (1)-(18) with \label/\eqref, Tables 1-13 and Figures 1-4 with \caption/\ref, bibliography as thebibliography with \cite{refN}. All labels checked against the Markdown numbering (35 labels, 0 mismatches), 22 pages, no undefined references.
- When the venue template (HYDRO INTERNATIONAL or the journal) is known, move the body into that template; the conversion script only needs a new preamble.
- After every phase: update master_draft.md, regenerate PDF and LaTeX, send both to the user.

## PROGRESS NOTE (2026-10-25, late): LATEX NOW IN IEEE FORMAT
- The user wants the IEEE conference look of their corruption-risk paper (two columns, IEEEtran, Roman numeral sections, three author blocks, Abstract plus Index Terms). latex/main.tex now uses \documentclass[conference]{IEEEtran}. Compiled locally: 14 pages, 0 errors, 35 labels match.
- References in master_draft.md rewritten in IEEE style ([n] A. Author, "Title," Journal, vol., no., pp., year). [17], [19], [20], [21] still carry [CONFIRM ...]. Index Terms line added under the abstract in master_draft.md.
- Author block: only the user is filled in (Santhosh Kumar D, SRM, sd8663@srmist.edu.in). Co-author and advisor blocks are placeholders; ask the user who the authors of THIS paper are before filling.
- Wide equations (5, 11, 14, 16, 17) are stacked in \begin{gathered} for the two-column layout (done in make_latex.py). Tables with 4+ columns and all figures are full width (table*, figure*).
- The venue's own template, if HYDRO INTERNATIONAL gives one, overrides this.

## PROGRESS NOTE (2026-10-30): PHASE G PAPER SECTION WRITTEN
- Paper: Methodology > Transformer Encoder added (self-attention with 4 heads, layer structure and output, checks, limitations). Eqs. (19) heads, (20) multi-head, (21) pre-norm layer, (22) feed-forward, (23) h_target = H_n[7,:]. Tables 14 (size of the temporal branch, 67,520 parameters) and 15 (checks). Next equation (24), next table 16, next figure 5 (architecture).
- Numbers (all from transformer/g_report.md): numpy hand check 6.2e-07; attention vs numpy softmax 5.9e-08; row sums within 1.8e-07; CPU vs GPU 7.2e-07; shuffle of first 7 readings without PE 4.8e-07, with PE 0.123; 28 parameter tensors with non-zero gradient; fused eval kernel differs 2.8e-04 (switched off); 67,520 parameters about 116 per training window (over-fitting risk stated in the paper).
- Untested design choices stated in the paper: pre-norm, last position output, two layers; last position vs mean pooling is a Phase L ablation.
- Outputs rebuilt: Groundwater_GNN_Transformer_Paper.pdf (single column, 22 pages) and latex/main.pdf (IEEE, 15 pages). latex/make_latex.py: wide equations now 5, 11, 14, 16, 17, 19, 21, 22. Backup: archive/master_draft_before_phaseG.md.
- TEACHING RULE (user request 2026-10-30): explanations must be SMALL or MEDIUM (about 10 to 12 lines per step): one real example, one formula, what the code does in 1 or 2 lines, one-line mental model, one question. Do not dump parameter counts, all checks and limitations in the explanation; they are in the paper and reports. The user asked to understand Phase G by reading the paper section paragraph by paragraph. Open vivas: C2, C3, C4, D5, E overview, E1, E3, F1, F2, F3, G.
