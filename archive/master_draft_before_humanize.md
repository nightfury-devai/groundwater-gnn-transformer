# Spatio-Temporal GNN-Transformer Framework for Groundwater Level Forecasting in the Cauvery Delta Alluvial Aquifer, India

*[Author names and affiliations to be added before submission]*

## Abstract

[ABSTRACT, TO BE WRITTEN LAST, AFTER ALL RESULTS EXIST]

## Introduction

Groundwater is the main source of water for much of the Cauvery delta in Tamil Nadu [1], and the pressure on it is documented. The Central Ground Water Board (CGWB) classifies Thanjavur district as over-exploited and Thiruvarur as semi-critical, with extraction at 70 to 90% of the water that is replenished each year. Nagapattinam is missing from the same assessment because every assessed groundwater unit there is saline and unusable [1]. Two processes drive this. Heavy pumping for agriculture lowers the water table, and seawater from the Bay of Bengal moves inland to fill the space.

The three districts sit on one continuous alluvial aquifer, made of sand, silt and clay laid down by the Cauvery. A well in this aquifer is not an isolated point. It draws on a connected body of water, so a change recorded at one well says something about its neighbours. A forecast that treats each well as a separate time series discards that information.

Most published work does exactly that. In Indian studies in particular, each well is usually given its own recurrent model, such as an LSTM, and nothing passes between wells (see Related Work). Some studies do build a graph over the wells, but they pair it with a recurrent or convolutional temporal model, not an attention-based one. Others use a Transformer for the temporal side and ignore the spatial layout. We found one study that joins a real spatial graph with a real Transformer encoder. Hua et al. [2] apply it to eight wells in California, using groundwater levels only, with no rainfall, temperature or other weather input.

This paper applies a graph neural network combined with a Transformer to the Cauvery delta aquifer. A graph convolutional encoder represents the connection between wells. A Transformer encoder models the time pattern at each well, and the two are combined before the final prediction. The inputs are CGWB water-level records for 53 wells together with rainfall, temperature, humidity and evapotranspiration, so the model can use the monsoon recharge and dry-season depletion that control this aquifer. To our knowledge, this is the first application of spatio-temporal graph and Transformer modelling to alluvial aquifer dynamics in the Cauvery delta.

The main limitation is the data. CGWB wells are measured quarterly, so each well has far fewer readings than the daily or monthly sensor records used in some earlier studies. Our advantage is the size of the network, not the length of each record, and we report results with that in mind.

The contributions of this paper are:

1. A hybrid architecture that models well-to-well connectivity with a graph and the time pattern with a Transformer.
2. An application to a 53-well network that combines CGWB groundwater records with IMD and NASA POWER weather data. The network is more than six times larger than the only earlier graph and Transformer study for groundwater, which used 8 wells [2].
3. An application to a connected alluvial delta aquifer, where the geology gives the spatial graph a physical basis.

## Related Work

We reviewed thirteen studies that forecast groundwater levels with machine learning. Table 1 summarises how each one treats space and time. The review is organised by what the study does about the spatial relationship between wells.

**Table 1. Summary of the thirteen reviewed studies and the present work.**

| Study | Region | Wells | Spatial treatment | Temporal model |
|---|---|---|---|---|
| Thakur et al. [3] | Himachal Pradesh, India | 9 | None | PCA-DWT-LSTM, one per well |
| Dey and Singh [12] | Chhattisgarh, India | 267 | None | Two-layer fully connected network |
| Sankaran et al. [13] | Tiruppur, India | Pooled into one signal | None | ICEEMDAN-VMD with SMA-optimised LSTM, CNN-LSTM |
| Nolte et al. [14] | Nine regions, including India | More than 1,800 | Trainable well embedding | Single-layer LSTM |
| Wunsch et al. [4] | Upper Rhine Graben, Germany | 17 | None | NARX, LSTM, 1D-CNN |
| Zhuang et al. [5] | Jinan, China | 27 | K-nearest-neighbour graph, GraphSAGE | Multi-branch GRU with cross-attention fusion |
| Wang et al. [6] | Xiangtan, China | 16 | Delaunay graph, graph convolution | LSTM |
| Wu et al. [7] | North China Plain | 15 | Five fused adjacency matrices | GRU with an attention layer |
| Taccari et al. [8] | Overbetuwe, Netherlands | 213 | Nearest-neighbour graph | Dilated gated convolutions |
| Zhu et al. [9] | Guilin, China (karst) | 3 sites | None | Nine models, including a Transformer |
| Liu et al. [10] | Kuitun basin, China | 3 | None | TCN, Transformer, LSTM chain |
| Rodrigues et al. [11] | Urucuia aquifer, Brazil | 8 (plus 26 other sensors) | None in the model | Transformer and wavelet Crossformer with a gate |
| Hua et al. [2] | California, USA | 8 | Distance-weighted graph attention | Transformer with time-interval bias |
| **This work** | **Cauvery delta, India** | **53** | **Graph convolution** | **Transformer** |

### Models that treat each well on its own

Indian groundwater forecasting has mostly treated each well as a separate problem, or merged a whole district into one signal. Thakur et al. [3] applied a PCA-DWT-LSTM pipeline to nine wells in Himachal Pradesh, using rainfall, evapotranspiration, soil moisture and temperature from several sources, which is close to the data mix used here. Each well had its own LSTM, so nothing learned at one well could help at another. Dey and Singh [12] forecast 267 wells in Chhattisgarh with a two-layer fully connected network. It has no recurrent unit, no spatial term and no attention, and it is the simplest model among the thirteen. The authors name attention-based deep learning as a direction they did not pursue.

Sankaran et al. [13] worked in Tiruppur, in the same state as this study and the closest region among the thirteen. They pooled several monitoring wells into one regional signal, then fitted decomposition-based LSTM and CNN-LSTM models. Pooling removes whatever spatial structure the network had, and the authors list well-specific spatial modelling and Transformer models as directions left open. Nolte et al. [14] went the other way and trained one model on more than 1,800 wells from nine regions, 200 of them in India. Each well is represented only by a trainable vector, not by links to its neighbours, and the model is a single-layer LSTM. The authors state that comparison with graph and Transformer models remains an open question.

The pattern is not limited to India. Wunsch et al. [4] compared NARX, LSTM and 1D-CNN models on seventeen wells in the Upper Rhine Graben and found NARX generally the best. Like most benchmarks of this kind, each well was modelled separately. Across these studies, the temporal model gets refined and compared in detail while the layout of the wells is ignored or collapsed.

### Models that use a graph over the wells

A smaller group of studies builds an explicit graph, but the temporal side is recurrent or convolutional. Zhuang et al. [5] linked 27 wells in Jinan with a K-nearest-neighbour graph processed by GraphSAGE. The graph output goes into a three-scale GRU, and a cross-attention layer merges the branches. That attention blends the model's own branches. It does not replace the GRU as the sequence encoder, so the design is a graph plus a recurrent model, not a graph plus a Transformer.

Wang et al. [6] built a Delaunay graph over 16 wells in Xiangtan, with edge weights from distance and terrain similarity, and used an LSTM with no attention. Wu et al. [7] combined five adjacency matrices (distance, rainfall correlation, hydraulic gradient, water-level correlation and sub-basin membership) across 15 wells on the North China Plain. Their temporal model is a GRU with a shallow attention layer on top, not a stacked self-attention encoder. Taccari et al. [8] connected 213 piezometers in the Overbetuwe polder and used dilated gated convolutions. The model beat a MODFLOW simulation at most test piezometers, but the setting is a flat polder controlled by river stage and pumping, which differs from a monsoon-fed alluvial delta.

### Models that use attention without a spatial structure

The reverse pattern is just as common. Zhu et al. [9] benchmarked nine models, including a four-head Transformer, at three karst sites in Guilin. They kept the sites independent on purpose, so no graph or cross-well mechanism was involved. Liu et al. [10] chained a temporal convolutional network, a Transformer encoder and an LSTM for three wells in the Kuitun basin. The middle stage does use multi-head self-attention, but each well is modelled alone. Rodrigues et al. [11] ran a standard Transformer and a wavelet Crossformer in parallel, merged by a learned gate, on eight wells and twenty-six other sensors in Brazil's Urucuia aquifer. They analysed cross-correlation between wells afterwards, but that analysis sits outside the forecasting model and never enters it as a graph.

### The closest prior work

Of the thirteen studies, only one combines a spatial graph with a Transformer encoder. Hua et al. [2] built a distance-weighted graph attention network over eight wells in California and paired it with a Transformer that adds a time-interval bias to attention, designed for irregular sampling and sudden recharge events. It uses groundwater levels alone, with no rainfall, temperature or other weather variable. The network is also much smaller than ours: eight wells against 53, a difference of more than six times. The present study differs in region, in network size, in the inclusion of weather inputs and in the aquifer type it addresses.

## Study Area and Data

### Study Area

The study area covers three districts of Tamil Nadu, Thanjavur, Thiruvarur, and Nagapattinam, which together form the core of the Cauvery delta. The 53 observation wells used here lie between 10.26° and 11.29° N and between 78.95° and 79.86° E (Figure 1). The median distance from a well to its nearest neighbour is 6.7 km (mean 7.0 km, maximum 19.1 km), and the largest separation between any two wells is 125.3 km. Thanjavur contributes 26 wells, Nagapattinam 16, and Thiruvarur 11. Of the 1,232 water-level readings, 1,098 are tagged by CGWB to the Cauvery basin (Cauvery Lower sub-basin) and the remaining 134 to the East Flowing Rivers Between Pennar and Kanyakumari basin.

The three districts sit on one continuous alluvial aquifer of sand, silt, and clay laid down by the Cauvery and its distributaries. Because the water-bearing material is a single connected body rather than a set of isolated fractures, as in the hard-rock terrain found elsewhere in Tamil Nadu, the water level at one well carries information about its neighbours. This is the physical basis for representing the wells as a graph.

The districts differ sharply in their groundwater condition. Thanjavur is classified as over-exploited and Thiruvarur as semi-critical, with extraction at 70 to 90% of annual replenishment, while Nagapattinam is left out of the standard CGWB district assessment because all of its assessed groundwater units are saline [1]. The Nagapattinam wells are retained in the network because they record the same aquifer system and the region's salinity problem, but readers should note that their depths reflect a coastal setting where seawater influence is a competing process to abstraction. In 2020 Nagapattinam was divided into Nagapattinam and Mayiladuthurai districts. The well records used here carry the district names assigned in the CGWB files, and no re-assignment of wells between the two successor districts was attempted.

![](figures/fig1_study_area.png)

**Figure 1.** Study area and the 53 CGWB observation wells, by district: Thanjavur (n = 26), Thiruvarur (n = 11), and Nagapattinam (n = 16). District outlines are taken from the boundary file used for the map, and the Nagapattinam outline includes the area that became Mayiladuthurai district in 2020. A few wells lie on or very close to a district line, so their colour may not match the polygon they sit in.

### Data Sources

Four datasets from three agencies were combined (Table 2). Groundwater depth is the forecasting target. Rainfall, temperature, humidity, and evapotranspiration are the meteorological drivers. All sources cover January 2013 onward. Groundwater readings end on 30 November 2024, and the weather records extend to 31 December 2025.

**Table 2. Datasets used in this study.**

| Variable | Provider | Spatial form | Records | Period |
|---|---|---|---|---|
| Depth to water level (m below ground) | CGWB, via the National Water Data Portal | 53 wells | 1,232 readings | 2013-01-01 to 2024-11-30 |
| Daily rainfall (mm) | IMD | 0.25° grid, 18 cells | 85,464 cell-days | 2013-01-01 to 2025-12-31 |
| Daily maximum and minimum temperature (°C) | IMD | 1° grid, 9 cells | 42,732 cell-days | 2013-01-01 to 2025-12-31 |
| Daily relative humidity (%) and evapotranspiration (mm/day) | NASA POWER | Point value at each well | 251,644 well-days | 2013-01-01 to 2025-12-31 |

#### Groundwater levels

The target is depth to water level in metres below ground level, so a larger value means a deeper water table. The readings come from the CGWB quarterly manual measurement programme. The first 838 readings (to January 2021) came from CGWB records for the 53 wells. A further 394 readings, dated 10 January 2021 to 30 November 2024 and covering 34 of the 53 wells, were added from the CGWB Tamil Nadu quarterly files on the National Water Data Portal. Well names were matched exactly; wells whose names did not appear in the newer file under the same name were not matched approximately, so 19 wells have no readings after January 2021. Re-deriving the earlier period from the new pipeline reproduced all 838 original merged rows without a single difference. The older records are dated one day before the portal's dates; after shifting by one day, 789 of 790 overlapping values were identical, and since readings are later assigned to calendar quarters, this offset matters for only one reading (Vallam, 1 January 2023).

The record is uneven across wells. The number of readings per well ranges from 1 to 43 with a median of 26; 42 wells have at least 9 readings and 33 have at least 17. By district the readings are 552 (Thanjavur), 471 (Nagapattinam), and 209 (Thiruvarur). The number of readings per calendar year varies from 61 (2020) to 142 (2013). Depth to water ranges from -0.90 to 21.32 m, with a median of 3.66 m and a mean of 4.67 m (standard deviation 4.03 m). Mean depth is 5.43 m in Nagapattinam, 4.58 m in Thanjavur, and 3.16 m in Thiruvarur (Table 3). Three readings from a single well (Allankottai) are negative (-0.74 m on 10 January 2021, -0.77 m on 20 January 2023, and -0.90 m on 30 November 2024). They are retained in the dataset; their cause was not established from the records. Of the 1,232 readings, 1,211 fall in distinct well-quarter slots, and 21 are second readings from the same well in the same quarter, which are resolved in preprocessing.

**Table 3. Depth to water level by district (m below ground level).**

| District | Wells | Readings | Mean | Std. dev. | Min | Median | Max |
|---|---|---|---|---|---|---|---|
| Thanjavur | 26 | 552 | 4.58 | 3.26 | 0.03 | 4.20 | 20.00 |
| Nagapattinam | 16 | 471 | 5.43 | 5.17 | 0.00 | 3.47 | 21.32 |
| Thiruvarur | 11 | 209 | 3.16 | 1.95 | -0.90 | 2.85 | 10.68 |
| All | 53 | 1,232 | 4.67 | 4.03 | -0.90 | 3.66 | 21.32 |

#### Rainfall and temperature

Rainfall and temperature were taken from the gridded products of the India Meteorological Department, the 0.25° daily rainfall dataset [15] and the 1° daily temperature dataset [16]. The study area is covered by 18 rainfall cells and 9 temperature cells. The temperature files use a fill code (99.9) for missing cells, which was removed before use because it otherwise inflated the averages.

IMD rainfall was kept after comparing it with the alternatives available for this period. Sundaram and Radhakrishnan [17] evaluated APHRODITE, IMD, ERA5, and CHIRPS against gauge records in the Cauvery Delta for 1986 to 2005. They rated IMD as moderately reliable and suitable for operational use. APHRODITE ranked highest in that study but its current release covers only 1998 to 2015, so it cannot reach the readings used here. A separate evaluation of eight satellite and reanalysis products against gauges in four Tamil Nadu districts found ERA5-Land best in Tiruchirappalli, the gauge location nearest to our districts, with MSWEP a close second [18]. IMD was not among the products tested there, so that study does not rank IMD against ERA5-Land. We therefore treat IMD rainfall as adequate, not optimal, and do not claim it outperforms the alternatives.

#### Humidity and evapotranspiration

Daily relative humidity and evapotranspiration were obtained from NASA's Prediction of Worldwide Energy Resources (POWER) service [19] at the coordinates of each well. POWER reports evapotranspiration as an energy flux (MJ m⁻² day⁻¹), which was converted to a water depth in mm day⁻¹ by dividing by 2.45, the latent heat of vaporization in MJ kg⁻¹. The divisor was confirmed against the service metadata. POWER data are derived from reanalysis, and the comparison in [18] cautions that MERRA-2-based products tend to overestimate without bias correction. We did not apply any bias correction to humidity or evapotranspiration, and no regional validation of these two variables against ground measurements was carried out. POWER has been used for Tamil Nadu climate analysis in published work [20], which establishes its use in the region but not its accuracy for these two variables.

### Data Integration

The four sources differ in spatial form and in time step. Groundwater readings are irregular point observations, rainfall and temperature sit on two different regular grids, and humidity and evapotranspiration are continuous daily series at the wells. They were joined into one table with one row per well reading in two steps.

**Spatial matching.** Each well was assigned its nearest rainfall cell and, separately, its nearest temperature cell, because the two grids do not coincide. Distance was computed with the haversine formula,

$$d = 2R \arcsin\sqrt{\sin^2\!\left(\tfrac{\Delta\varphi}{2}\right) + \cos\varphi_1 \cos\varphi_2 \sin^2\!\left(\tfrac{\Delta\lambda}{2}\right)} \qquad\text{(1)}$$

where φ is latitude, λ is longitude, and R = 6371 km. The matching is shown in Figure 2. Matching rainfall and temperature to wells independently was necessary: an earlier attempt that joined them by exact coordinates left 94% of the temperature values empty. The median well-to-cell distance is 11.1 km for rainfall (mean 10.5 km, maximum 20.7 km), and 14 of the 18 rainfall cells are the nearest cell of at least one well. For temperature the median distance is 44.4 km (mean 43.0 km, maximum 66.0 km), and only 3 of the 9 cells are used. Temperature is therefore nearly uniform across neighbouring wells in this dataset, and the spatial differences between wells that the graph can exploit come mainly from the groundwater record itself and from rainfall, not from temperature.

![](figures/fig2_data_integration.png)

**Figure 2.** Nearest-cell matching of the 53 wells to (a) the 18 IMD rainfall cells (0.25 degree) and (b) the 9 IMD temperature cells (1 degree). Shaded cells are the nearest cell for at least one well; dashed cells are not used. Lines join each well to the centre of its matched cell. Temperature cells T1, T2, T3, T8, and T9 fall outside the plotted extent.

**Temporal aggregation.** A single day's weather does not explain a water level measured on that day, so each reading is paired with trailing-window summaries of the weather before it. For well i with a reading on day t,

$$R_{i,t} = \frac{1}{180}\sum_{d=t-179}^{t} r_{i,d} \qquad\text{(2)}$$

which is the mean daily rainfall over 180 days (mm day⁻¹),

$$V_{i,t} = \frac{1}{90}\sum_{d=t-89}^{t} v_{i,d}, \qquad v \in \{T_{\max},\, T_{\min},\, H\} \qquad\text{(3)}$$

for maximum temperature, minimum temperature and relative humidity,

$$E_{i,t} = \sum_{d=t-89}^{t} e_{i,d} \qquad\text{(4)}$$

which is evapotranspiration summed over 90 days (mm).

Rainfall is averaged over a longer window than the other variables because infiltration reaches the water table with a delay. Temperature and humidity describe conditions, so they are averaged; evapotranspiration is an accumulated loss, so it is summed. Windows are computed from the days available. Because the weather series begins on 1 January 2013, readings taken in January 2013 have shorter windows: 51 readings have fewer than 90 days of history and 88 have fewer than 180 days of rainfall history. No reading after January 2013 is affected. For the 90-day evapotranspiration sum the shortened windows give partial sums and therefore underestimate the true accumulation for those readings.

The integrated table has 1,232 rows and no missing values. Table 4 gives the summary statistics of the model inputs, and Table 5 gives the correlation of each input with groundwater depth.

**Table 4. Summary statistics of the integrated dataset (n = 1,232).**

| Variable | Mean | Std. dev. | Min | Max |
|---|---|---|---|---|
| Rainfall, 180-day mean (mm day⁻¹) | 3.20 | 2.23 | 0.00 | 10.33 |
| Maximum temperature, 90-day mean (°C) | 33.60 | 2.20 | 30.01 | 37.68 |
| Minimum temperature, 90-day mean (°C) | 24.95 | 1.33 | 21.91 | 27.00 |
| Relative humidity, 90-day mean (%) | 73.58 | 8.23 | 52.08 | 88.02 |
| Evapotranspiration, 90-day sum (mm) | 56.53 | 47.19 | 0.02 | 159.38 |
| Depth to water level (m) | 4.67 | 4.03 | -0.90 | 21.32 |

**Table 5. Correlation of each input with depth to water level.**

| Input | Spearman, all readings | Pearson, within-well deviations |
|---|---|---|
| Rainfall, 180-day mean | -0.34 | -0.36 |
| Maximum temperature, 90-day mean | 0.30 | 0.34 |
| Minimum temperature, 90-day mean | 0.24 | 0.26 |
| Relative humidity, 90-day mean | -0.38 | -0.41 |
| Evapotranspiration, 90-day sum | -0.32 | -0.35 |

*The within-well column subtracts each well's own mean from both the input and the depth and uses the 1,196 readings from the 42 wells with at least nine readings.*

The signs agree with the expected physics. More antecedent rainfall and humidity go with a shallower water table (smaller depth), and higher temperature goes with a deeper one. The negative sign for evapotranspiration should not be read as a causal effect: POWER evapotranspiration is highest in the wet season, so in this region it moves with the monsoon and shares its seasonal pattern with rainfall. These are descriptive correlations only, and the model, not this table, determines how much each input contributes to the forecast.

### Data Limitations

Four limitations bear on how the results should be read. First, the groundwater record is sparse in time: readings are quarterly, wells have between 1 and 43 readings, and many wells do not have long unbroken runs of consecutive quarters, so the number of usable training sequences is much smaller than the 1,232 readings suggest. The per-well record is far shorter than in studies built on daily or monthly sensor data, and the advantage of the present network lies in its number of wells and their spatial coverage, not in the volume of data per well. Second, 19 of the 53 wells have no reading after January 2021, so the later period is represented by 34 wells. Third, the coarse temperature grid contributes almost no spatial variation between wells, as shown above. Fourth, humidity and evapotranspiration are reanalysis-based values at the well location and have not been checked against local ground measurements. Telemetry data from the Tamil Nadu state network were examined as an additional source and not used: only 2 of the 53 wells have a telemetry station within 1 km, the sign convention of the depth values is not documented, and the record is not comparable with the manual readings at the one pair of stations that could be checked. We leave telemetry for future work.

## Methodology

The methodology has two parts. The first, given here, describes how the integrated table of readings becomes a set of supervised examples with a fixed chronological split. The model that learns from these examples is described in the part that follows.

### Data Preparation

Preparation has five steps: resolving duplicate readings, building the inputs and the target, fixing a chronological split, normalising, and cutting each well's record into fixed-length windows. The order matters. The split dates are fixed before normalisation, because the normalisation statistics must come from the training period alone. Table 6 shows how the number of samples changes at each step.

**Table 6. Number of readings and samples after each preparation step.**

| Step | Operation | Result |
|---|---|---|
| Integrated table | One row per CGWB reading | 1,232 readings, 53 wells |
| Duplicate resolution | Keep the later reading of each well-quarter | 1,211 readings |
| Inputs and target | Target is the next reading of the same well | 1,158 samples, 49 wells |
| Windowing, gap-tolerant (primary) | Eight most recent readings per sample | 840 windows, 42 wells |
| Windowing, strict (sensitivity check) | Eight consecutive quarters, target in the next quarter | 157 windows, 23 wells |

#### Resolving duplicate readings

Readings are assigned to calendar quarters, and 21 well-quarter slots held two readings from the same well (1,190 slots held one reading, and none held three or more). We kept the later reading of each pair and discarded the earlier one. We did not average the two values, because an average is a depth that was never measured, and we did not interpolate. The result is 1,211 readings in 43 distinct quarters, from the first quarter of 2013 to the last quarter of 2024. Five quarters contain no reading from any well (2015 Q3, 2016 Q1, 2020 Q2, 2020 Q3 and 2021 Q2), and we did not establish why.

The two readings of a pair can differ widely. The median absolute difference between the discarded and the kept reading is 0.56 m (mean 1.21 m). The largest is 9.35 m, at Melvastrachavdi in 2013 Q1, where the readings of 1 and 4 January 2013 are 9.78 m and 19.13 m. The kept value fits better with the well's other readings, which are 19.40, 17.10 and 20.00 m. The rule is applied the same way to every pair, and no pair was judged individually.

#### Inputs and forecasting target

Let well $i$ have readings on dates $t_{i,1} < t_{i,2} < \dots$. The input vector of the $k$-th reading holds the five weather summaries of Eqs. (2) to (4), evaluated at $t_{i,k}$, and the depth $d_{i,k}$ measured at that reading:

$$\mathbf{x}_{i,k} = \left[\, R_{i,t_k},\; T_{\max},\; T_{\min},\; H,\; E,\; d_{i,k} \,\right]^{\top} \in \mathbb{R}^{6}, \qquad y_{i,k} = d_{i,k+1} \qquad\text{(5)}$$

The target $y_{i,k}$ is the depth at the next reading of the same well. The weather windows end on the reading date, so no input uses information from after it. The last reading of each well has no next reading and therefore no target. This removes 53 readings and leaves 1,158 samples. Four wells have a single reading and give no sample, which leaves 49 wells.

The forecast horizon is the next reading, not a fixed length of time. Of the 1,158 samples, 839 (72.5%) have the next reading in the following quarter, 181 (15.6%) two quarters later, 86 (7.4%) three quarters later and 52 (4.5%) four or more, with the longest gap at 21 quarters. We therefore describe the task as next-reading forecasting. The elapsed time is kept as a separate quantity, described below.

#### Chronological split

Each sample is dated by its target. Samples are ordered by that date and divided with two cut dates, $\tau_1$ = 30 September 2022 and $\tau_2$ = 31 December 2023:

$$\mathrm{split}(s) = \begin{cases} \text{train}, & t_s \le \tau_1 \\ \text{validation}, & \tau_1 < t_s \le \tau_2 \\ \text{test}, & t_s > \tau_2 \end{cases} \qquad\text{(6)}$$

where $t_s$ is the target date of sample $s$. Both cut dates are quarter ends, chosen as the quarter ends at which the cumulative share of the windows defined below comes closest to 70% and 85%. Nothing is shuffled. A random split would place later readings of a well in training and earlier ones in testing, so the test score would no longer measure forecasting.

The training windows have targets from May 2015 to August 2022, the validation windows from November 2022 to November 2023, and the test windows from January 2024 to November 2024. Validation and test each cover about one year, and the test set has 112 windows from 32 wells, so test scores will carry a noticeable sampling error.

#### Normalisation

Each of the six inputs is standardised as

$$z = \frac{v - \mu_f}{\sigma_f} \qquad\text{(7)}$$

where $\mu_f$ and $\sigma_f$ are the mean and the population standard deviation of feature $f$ over the 952 readings dated on or before $\tau_1$. The 259 later readings did not contribute (Table 7). The depth input and the depth target share one pair of statistics, so a prediction in standardised units returns to metres with

$$d = z\,\sigma_d + \mu_d \qquad\text{(8)}$$

and every error in this paper is reported in metres.

**Table 7. Normalisation statistics, fitted on the 952 readings dated on or before 30 September 2022.**

| Feature | Mean | Std. dev. |
|---|---|---|
| Rainfall, 180-day mean (mm day⁻¹) | 3.156 | 2.303 |
| Maximum temperature, 90-day mean (°C) | 33.631 | 2.223 |
| Minimum temperature, 90-day mean (°C) | 24.986 | 1.317 |
| Relative humidity, 90-day mean (%) | 73.105 | 8.498 |
| Evapotranspiration, 90-day sum (mm) | 53.291 | 46.317 |
| Depth to water level (m) | 4.751 | 3.984 |

Fitting on all 1,211 readings instead would have shifted the evapotranspiration mean by 3.2 mm and the humidity mean by 0.36 percentage points. These shifts are small, but they would carry information from the test period into training, and fitting on the training period costs nothing. Values outside the training range are not clipped. In the validation and test sets, at most 1.8% of any column fell outside the training range (the target depth in the test set), and no weather feature exceeded 1.4%.

#### Sliding windows

A window collects the most recent readings of one well. For the reading with index $k$ of well $i$,

$$\mathbf{W}_{i,k} = \left[\, \mathbf{x}_{i,k-7},\; \mathbf{x}_{i,k-6},\; \dots,\; \mathbf{x}_{i,k} \,\right] \in \mathbb{R}^{8 \times 6}, \qquad \text{target } y_{i,k} \qquad\text{(9)}$$

so a window needs at least eight readings of the well up to and including reading $k$. For each position we also store $g_{i,j}$, the number of whole quarters since the previous reading of that well, and the number of quarters from the last input to the target. A window never mixes readings from two wells.

We built two versions because the record is uneven. In the **gap-tolerant** version, the window is the eight most recent readings whatever their spacing, and the gaps are recorded. In the **strict** version, the eight readings must fall in eight consecutive quarters and the target must lie in the next quarter. The strict version has the cleanest meaning but, as Table 8 shows, it is small. We use the gap-tolerant version as the main dataset and the strict version as a sensitivity check.

**Table 8. Windows and wells per split (target dates by split are those of the gap-tolerant windows).**

| Split | Gap-tolerant windows (wells) | Strict windows (wells) | Target dates |
|---|---|---|---|
| Train | 581 (42) | 80 (16) | 2015-05-14 to 2022-08-30 |
| Validation | 147 (34) | 26 (13) | 2022-11-30 to 2023-11-10 |
| Test | 112 (32) | 51 (15) | 2024-01-10 to 2024-11-30 |
| All | 840 (42) | 157 (23) | |

In the gap-tolerant set, 652 windows (77.6%) have the target in the next quarter, 115 (13.7%) two quarters later, 53 (6.3%) three quarters later and 20 (2.4%) four or more. The eight input readings span a median of 912 days (minimum 590, maximum 2,938), against a median of 639 days for the strict windows. In 42 gap-tolerant windows, the first input is the first reading of its well, so its gap is undefined and is left empty.

The strict set is thin in time and place. Its training windows have targets only between May 2018 and January 2020. The five quarters without any reading break every run of consecutive quarters that crosses them. No strict window has a Thiruvarur well in training or test, and the three Thiruvarur wells with strict windows appear only in validation.

To give later results a floor, we scored the simplest forecast, which repeats the last observed depth. On the gap-tolerant windows its mean absolute error is 0.888 m on validation and 1.347 m on test (root mean squared error 1.338 m and 1.826 m). A learned model is useful only if it improves on these values.

Three readings at Allankottai are below zero in the source data. They are kept as they are. They appear as the target of 3 gap-tolerant windows and among the inputs of 10, and a check without them is planned for the robustness experiments.

#### Leakage checks

Four conditions were checked by assertions in the code. Every target date is later than the last input date of its window. Every training target is dated on or before $\tau_1$, and every validation and test target after it. The normalisation statistics, recomputed from the readings dated on or before $\tau_1$, equal the stored ones. No window contains readings of two wells. Running the pipeline again reproduces every output file exactly.

The final dataset is therefore 840 windows, each with an $8 \times 6$ block of standardised inputs, the quarter gaps and a target depth in metres, of which 581 train the model, 147 are used to select it and 112 are held back for the final test (Figure 3).

![](figures/figC_split_and_windows.png)

**Figure 3.** Chronological split and the structure of a window. (a) Number of gap-tolerant windows per target quarter, coloured by split. Dashed lines mark the cut dates, 30 September 2022 and 31 December 2023. Quarters without a bar have no window, and five of them have no reading from any well. (b) One window at Keelvalur. The first eight readings of the well (filled points) are the input and the next reading (open circle) is the target. Grey points are later readings. The numbers along the bottom give the quarters between consecutive readings.

## References

[1] The News Minute. (2025, March 4). Tamil Nadu's paddy paradox: Cauvery delta turns on the taps even as wells run dry. https://www.thenewsminute.com/tamil-nadu/tamil-nadus-paddy-paradox-cauvery-delta-turns-on-the-taps-even-as-wells-run-dry

[2] Hua, J., Hu, X., Ge, W., Shen, Y., & Kang, C. (2025). Graph-embedded time-interval-aware transformer for event-driven groundwater level forecasting. *AIP Advances*, 15, 105226.

[3] Thakur, A., Chandel, A. S., & Shankar, V. (2025). Prediction of groundwater levels using a long short-term memory (LSTM) technique. *Journal of Hydroinformatics*, 27(1), 51.

[4] Wunsch, A., Liesch, T., & Broda, S. (2021). Groundwater level forecasting with artificial neural networks: a comparison of long short-term memory (LSTM), convolutional neural networks (CNNs), and non-linear autoregressive networks with exogenous input (NARX). *Hydrology and Earth System Sciences*, 25, 1671-1687.

[5] Zhuang, C., Cui, L., & Cui, Y. (2025). Enhancing groundwater level prediction with a hybrid deep learning model in Jinan City, China. *Scientific Reports*, 15, 44535.

[6] Wang, L., et al. (2024). A groundwater level spatiotemporal prediction model based on graph convolutional networks with a long short-term memory. *Journal of Hydroinformatics*, 26(11), 2962-2979.

[7] Wu, Y., et al. (2025). Forecasting groundwater level by characterizing multiple spatial dependencies of environmental factors using graph-based deep learning. *JGR: Machine Learning and Computation*, 2, e2024JH000520.

[8] Taccari, M. L., Wang, H., Nuttall, J., Chen, X., & Jimack, P. K. (2024). Spatial-temporal graph neural networks for groundwater data. *Scientific Reports*, 14, 24564.

[9] Zhu, Q., Zhu, Y., Niu, J., et al. (2026). Benchmarking machine learning and deep learning models for groundwater level prediction in karst aquifers. *Water*, 18(8), 939.

[10] Liu, Y., et al. (2025). Groundwater level prediction using a hybrid TCN-Transformer-LSTM model and multi-source data fusion: a case study of the Kuitun River Basin, Xinjiang. *Sustainability*, 17(19), 8544.

[11] Rodrigues, V. H. S. R., et al. (2023). Wavelet gated multiformer for groundwater time series forecasting. *Scientific Reports*, 13, 12726.

[12] Dey, S., & Singh, V. (2025). Forecasting of groundwater level variation under changing climate in Chhattisgarh state using deep learning technique. *Hydrology Research*, 56(3), 279-296.

[13] Sankaran, S., Krishnan, R., & Chidambaram, S. (2026). Assessing and forecasting groundwater resources in the context of climate change using AI techniques for the industry zones in Tiruppur, India. *World*, 7(5), 79.

[14] Nolte, C., et al. (2025). Multi-site deep learning for groundwater level prediction across global datasets. *Journal of Hydroinformatics*, 27(10), 1632.

[15] Pai, D. S., Rajeevan, M., Sreejith, O. P., Mukhopadhyay, B., & Satbha, N. S. (2014). Development of a new high spatial resolution (0.25° × 0.25°) long period (1901-2010) daily gridded rainfall data set over India and its comparison with existing data sets over the region. *Mausam*, 65(1), 1-18.

[16] Srivastava, A. K., Rajeevan, M., & Kshirsagar, S. R. (2009). Development of a high resolution daily gridded temperature data set (1969-2005) for the Indian region. *Atmospheric Science Letters*, 10(4), 249-254.

[17] Sundaram, & Radhakrishnan. (2025). Evaluating the accuracy and applicability of precipitation datasets for sustainable agro-hydrological and climatic planning in Tamil Nadu's Cauvery Delta. *Journal of Earth System Science*. https://doi.org/10.1007/s12040-025-02690-3 [CONFIRM INITIALS, VOLUME, ARTICLE NUMBER]

[18] Saravanan, A., Karthe, D., Ramalingam, S., & Schütze, N. (2025). Evaluation of remote-sensing- and reanalysis-based precipitation products for agro-hydrological studies in the semi-arid tropics of Tamil Nadu. *Hydrology and Earth System Sciences*, 29(19), 4847-4870. https://doi.org/10.5194/hess-29-4847-2025

[19] NASA Langley Research Center. Prediction of Worldwide Energy Resources (POWER) Project. https://power.larc.nasa.gov [CONFIRM PREFERRED CITATION AND ACCESS DATE]

[20] Drought monitoring in Tamil Nadu using the SPEI index with NASA POWER data, 1991-2024. *Plant Science Today*. https://horizonepublishing.com/journals/index.php/PST/article/view/4653 [CONFIRM AUTHORS, TITLE, YEAR]
