# Spatio-Temporal GNN-Transformer Framework for Groundwater Level Forecasting in the Cauvery Delta Alluvial Aquifer, India

*[Author names and affiliations to be added before submission]*

## Abstract

[ABSTRACT, TO BE WRITTEN LAST, AFTER ALL RESULTS EXIST]

*Index Terms:* groundwater level forecasting, graph convolutional network, Transformer, spatio-temporal modelling, alluvial aquifer, Cauvery delta

## Introduction

Much of the Cauvery delta in Tamil Nadu depends on groundwater [1], and the strain on it is well documented. The Central Ground Water Board (CGWB) classifies Thanjavur district as over-exploited and Thiruvarur as semi-critical, with extraction at 70 to 90% of the water replenished each year. Nagapattinam does not appear in the same assessment, because every groundwater unit assessed there is saline and unusable [1]. Two processes are behind this. Heavy pumping for agriculture lowers the water table, and seawater from the Bay of Bengal moves inland to fill the space that is left.

The three districts lie over one continuous alluvial aquifer of sand, silt and clay laid down by the Cauvery. A well here is not an isolated point. It draws on a connected body of water, so what is recorded at one well says something about its neighbours, and a forecast that treats every well as a separate time series throws that information away.

Most published work does exactly that. Indian studies in particular tend to give each well its own recurrent model, an LSTM for example, so nothing passes between wells (see Related Work). A smaller group builds a graph over the wells but pairs it with a recurrent or convolutional model for time, not an attention-based one. Others use a Transformer for the temporal side and ignore where the wells are. We found only one study that combines a spatial graph with a Transformer encoder. Hua et al. [2] applied it to eight wells in California, using groundwater levels alone, with no rainfall, temperature or other weather input.

This paper applies a graph neural network combined with a Transformer to the Cauvery delta aquifer. A graph convolutional encoder captures the connection between wells, a Transformer encoder models the time pattern at each well, and the two are fused before the final prediction. The inputs are CGWB water-level records for 53 wells together with rainfall, temperature, humidity and evapotranspiration, which lets the model see the monsoon recharge and dry-season depletion that govern this aquifer. To our knowledge, this is the first application of spatio-temporal graph and Transformer modelling to alluvial aquifer dynamics in the Cauvery delta.

The main limitation is the data. CGWB wells are measured quarterly, so each well has far fewer readings than the daily or monthly sensor records used in some earlier studies. What we have is a larger network, not longer records, and we interpret the results with that in mind.

The contributions of this paper are:

1. A hybrid architecture that models well-to-well connectivity with a graph and the time pattern with a Transformer.
2. An application to a 53-well network that combines CGWB groundwater records with IMD and NASA POWER weather data. The network is more than six times larger than the only earlier graph and Transformer study for groundwater, which used 8 wells [2].
3. An application to a connected alluvial delta aquifer, where the geology gives the spatial graph a physical basis.

## Related Work

We reviewed thirteen studies that forecast groundwater levels with machine learning. Table 1 summarises how each treats space and time, and the discussion below is organised by what a study does about the spatial relationship between wells.

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

Indian groundwater forecasting has mostly treated each well as its own problem, or merged a whole district into one signal. Thakur et al. [3] applied a PCA-DWT-LSTM pipeline to nine wells in Himachal Pradesh, using rainfall, evapotranspiration, soil moisture and temperature from several sources, a data mix close to the one used here. Each well had its own LSTM, so nothing learned at one well could help at another. Dey and Singh [12] forecast 267 wells in Chhattisgarh with a two-layer fully connected network. It has no recurrent unit, no spatial term and no attention, which makes it the simplest model among the thirteen, and the authors name attention-based deep learning as a direction they did not pursue.

Sankaran et al. [13] worked in Tiruppur, in the same state as this study and the closest region among the thirteen. They pooled several monitoring wells into one regional signal and fitted decomposition-based LSTM and CNN-LSTM models. Pooling erases whatever spatial structure the network had, and the authors themselves list well-specific spatial modelling and Transformer models as directions left open. Nolte et al. [14] went the opposite way and trained one model on more than 1,800 wells from nine regions, 200 of them in India. Each well is represented only by a trainable vector, with no links to its neighbours, and the model is a single-layer LSTM. The authors state that a comparison with graph and Transformer models remains an open question.

The pattern is not confined to India. Wunsch et al. [4] compared NARX, LSTM and 1D-CNN models on seventeen wells in the Upper Rhine Graben and found NARX generally the best, and like most benchmarks of this kind they modelled each well separately. Across these studies the temporal model is refined and compared in detail, while the layout of the wells is ignored or collapsed.

### Models that use a graph over the wells

A smaller group of studies builds an explicit graph, but their temporal side is recurrent or convolutional. Zhuang et al. [5] linked 27 wells in Jinan with a K-nearest-neighbour graph processed by GraphSAGE. The graph output feeds a three-scale GRU, and a cross-attention layer merges the branches. That attention blends the model's own branches and does not replace the GRU as the sequence encoder, so the design is a graph plus a recurrent model, not a graph plus a Transformer.

Wang et al. [6] built a Delaunay graph over 16 wells in Xiangtan, with edge weights from distance and terrain similarity, and paired it with an LSTM without attention. Wu et al. [7] combined five adjacency matrices (distance, rainfall correlation, hydraulic gradient, water-level correlation and sub-basin membership) across 15 wells on the North China Plain. Their temporal model is a GRU with a shallow attention layer on top, not a stacked self-attention encoder. Taccari et al. [8] connected 213 piezometers in the Overbetuwe polder and used dilated gated convolutions. Their model beat a MODFLOW simulation at most test piezometers, but a flat polder controlled by river stage and pumping is a different setting from a monsoon-fed alluvial delta.

### Models that use attention without a spatial structure

The reverse pattern is just as common. Zhu et al. [9] benchmarked nine models, including a four-head Transformer, at three karst sites in Guilin. They kept the sites independent on purpose, so no graph or cross-well mechanism was involved. Liu et al. [10] chained a temporal convolutional network, a Transformer encoder and an LSTM for three wells in the Kuitun basin. The middle stage uses multi-head self-attention, but each well is modelled alone. Rodrigues et al. [11] ran a standard Transformer and a wavelet Crossformer in parallel, merged by a learned gate, on eight wells and twenty-six other sensors in Brazil's Urucuia aquifer. They did analyse cross-correlation between wells, but only afterwards and outside the forecasting model, so it never enters the model as a graph.

### The closest prior work

Of the thirteen studies, only one combines a spatial graph with a Transformer encoder. Hua et al. [2] built a distance-weighted graph attention network over eight wells in California and paired it with a Transformer that adds a time-interval bias to attention, a design aimed at irregular sampling and sudden recharge events. It uses groundwater levels alone, with no rainfall, temperature or other weather variable, and its network is much smaller than ours: eight wells against 53, a difference of more than six times. The present study differs in region, network size, weather inputs and the type of aquifer it addresses.

## Study Area and Data

### Study Area

The study area covers three districts of Tamil Nadu, Thanjavur, Thiruvarur and Nagapattinam, which form the core of the Cauvery delta. The 53 observation wells lie between 10.26° and 11.29° N and between 78.95° and 79.86° E (Figure 1). A well's nearest neighbour is a median 6.7 km away (mean 7.0 km, maximum 19.1 km), and the largest separation between two wells is 125.3 km. Thanjavur contributes 26 wells, Nagapattinam 16 and Thiruvarur 11. CGWB tags 1,098 of the 1,232 readings to the Cauvery Lower sub-basin and 134 to the East Flowing Rivers Between Pennar and Kanyakumari basin.

The three districts sit on one continuous alluvial aquifer of sand, silt and clay laid down by the Cauvery and its distributaries. The water-bearing material is a single connected body, unlike the isolated fractures of the hard-rock terrain found elsewhere in Tamil Nadu, so the water level at one well carries information about its neighbours. That is the physical basis for representing the wells as a graph.

The districts differ in their groundwater condition. Thanjavur is classified as over-exploited and Thiruvarur as semi-critical, with extraction at 70 to 90% of annual replenishment, while Nagapattinam is left out of the standard CGWB district assessment because all of its assessed groundwater units are saline [1]. We keep the Nagapattinam wells because they record the same aquifer system, but their depths reflect a coastal setting where seawater influence competes with abstraction. Nagapattinam was divided into Nagapattinam and Mayiladuthurai in 2020, and we kept the district names of the CGWB files without re-assigning any well.

![](figures/fig1_study_area.png)

**Figure 1.** Study area and the 53 CGWB observation wells, by district: Thanjavur (n = 26), Thiruvarur (n = 11), and Nagapattinam (n = 16). District outlines are taken from the boundary file used for the map, and the Nagapattinam outline includes the area that became Mayiladuthurai district in 2020. A few wells lie on or very close to a district line, so their colour may not match the polygon they sit in.

### Data Sources

We combined four datasets from three agencies (Table 2). Groundwater depth is the forecasting target, and rainfall, temperature, humidity and evapotranspiration are the meteorological drivers. All sources start in January 2013. The groundwater readings end on 30 November 2024, and the weather records run to 31 December 2025.

**Table 2. Datasets used in this study.**

| Variable | Provider | Spatial form | Records | Period |
|---|---|---|---|---|
| Depth to water level (m below ground) | CGWB, via the National Water Data Portal | 53 wells | 1,232 readings | 2013-01-01 to 2024-11-30 |
| Daily rainfall (mm) | IMD | 0.25° grid, 18 cells | 85,464 cell-days | 2013-01-01 to 2025-12-31 |
| Daily maximum and minimum temperature (°C) | IMD | 1° grid, 9 cells | 42,732 cell-days | 2013-01-01 to 2025-12-31 |
| Daily relative humidity (%) and evapotranspiration (mm/day) | NASA POWER | Point value at each well | 251,644 well-days | 2013-01-01 to 2025-12-31 |

#### Groundwater levels

The target is depth to water level in metres below ground level, so a larger value means a deeper water table. The readings come from the CGWB quarterly manual measurement programme. The first 838 readings (to January 2021) came from CGWB records for the 53 wells, and we added 394 readings dated 10 January 2021 to 30 November 2024, covering 34 wells, from the CGWB Tamil Nadu quarterly files on the National Water Data Portal. Well names were matched exactly and never approximately, which is why 19 wells have no reading after January 2021. Re-deriving the earlier period with the new pipeline reproduced all 838 merged rows, and the one-day date offset between the two sources affects only one reading once readings are assigned to quarters (Vallam, 1 January 2023).

The record is uneven across wells. Readings per well range from 1 to 43 (median 26), 42 wells have at least 9 readings and 33 have at least 17, and readings per year vary from 61 (2020) to 142 (2013). Depth to water ranges from -0.90 to 21.32 m, with a median of 3.66 m and a mean of 4.67 m (standard deviation 4.03 m), and the mean differs by district (Table 3). Three readings from one well (Allankottai) are negative (-0.74, -0.77 and -0.90 m). We kept them, and their cause could not be established from the records. Of the 1,232 readings, 21 are second readings from the same well in the same quarter, which we resolve in preprocessing.

**Table 3. Depth to water level by district (m below ground level).**

| District | Wells | Readings | Mean | Std. dev. | Min | Median | Max |
|---|---|---|---|---|---|---|---|
| Thanjavur | 26 | 552 | 4.58 | 3.26 | 0.03 | 4.20 | 20.00 |
| Nagapattinam | 16 | 471 | 5.43 | 5.17 | 0.00 | 3.47 | 21.32 |
| Thiruvarur | 11 | 209 | 3.16 | 1.95 | -0.90 | 2.85 | 10.68 |
| All | 53 | 1,232 | 4.67 | 4.03 | -0.90 | 3.66 | 21.32 |

#### Rainfall and temperature

Rainfall and temperature come from the gridded products of the India Meteorological Department, the 0.25° daily rainfall dataset [15] and the 1° daily temperature dataset [16]. The study area is covered by 18 rainfall cells and 9 temperature cells. The temperature files mark missing cells with a fill code (99.9), which we removed before use because it otherwise inflated the averages.

We kept IMD rainfall after comparing it with the alternatives available for this period. Sundaram and Radhakrishnan [17] evaluated APHRODITE, IMD, ERA5 and CHIRPS against gauge records in the Cauvery Delta for 1986 to 2005 and rated IMD as moderately reliable and suitable for operational use. APHRODITE ranked highest, but its current release covers only 1998 to 2015 and cannot reach the readings used here. A separate evaluation of eight products against gauges in four Tamil Nadu districts found ERA5-Land best in Tiruchirappalli, the gauge location nearest to our districts [18], without testing IMD. We therefore treat IMD rainfall as adequate, not optimal, and do not claim it outperforms the alternatives.

#### Humidity and evapotranspiration

We obtained daily relative humidity and evapotranspiration from NASA's Prediction of Worldwide Energy Resources (POWER) service [19] at the coordinates of each well. POWER reports evapotranspiration as an energy flux (MJ m⁻² day⁻¹), which we converted to mm day⁻¹ by dividing by 2.45, the latent heat of vaporization in MJ kg⁻¹. POWER data are reanalysis-derived, and [18] cautions that MERRA-2-based products tend to overestimate without bias correction. We applied no bias correction and did not validate these two variables against ground measurements. POWER has been used for Tamil Nadu climate analysis [20], which shows it is used in the region but says nothing about its accuracy for these variables.

### Data Integration

The four sources differ in spatial form and in time step, so we joined them into one table with one row per well reading, in two steps.

**Spatial matching.** Each well was assigned its nearest rainfall cell and, separately, its nearest temperature cell, because the two grids do not coincide. Distances use the haversine formula,

$$d = 2R \arcsin\sqrt{\sin^2\!\left(\tfrac{\Delta\varphi}{2}\right) + \cos\varphi_1 \cos\varphi_2 \sin^2\!\left(\tfrac{\Delta\lambda}{2}\right)} \qquad\text{(1)}$$

where φ is latitude, λ is longitude and R = 6371 km. Figure 2 shows the matching. Matching rainfall and temperature to the wells independently was necessary, because joining them by exact coordinates left 94% of the temperature values empty. The median well-to-cell distance is 11.1 km for rainfall (maximum 20.7 km) and 44.4 km for temperature (maximum 66.0 km), and only 3 of the 9 temperature cells are used. Temperature is therefore nearly uniform across neighbouring wells, so the spatial differences the graph can exploit come mainly from the groundwater record itself and from rainfall.

![](figures/fig2_data_integration.png)

**Figure 2.** Nearest-cell matching of the 53 wells to (a) the 18 IMD rainfall cells (0.25 degree) and (b) the 9 IMD temperature cells (1 degree). Shaded cells are the nearest cell for at least one well; dashed cells are not used. Lines join each well to the centre of its matched cell. Temperature cells T1, T2, T3, T8, and T9 fall outside the plotted extent.

**Temporal aggregation.** A single day's weather does not explain a water level measured on that day, so each reading is paired with trailing-window summaries of the weather before it. For well i with a reading on day t,

$$R_{i,t} = \frac{1}{180}\sum_{d=t-179}^{t} r_{i,d} \qquad\text{(2)}$$

which is the mean daily rainfall over 180 days (mm day⁻¹),

$$V_{i,t} = \frac{1}{90}\sum_{d=t-89}^{t} v_{i,d}, \qquad v \in \{T_{\max},\, T_{\min},\, H\} \qquad\text{(3)}$$

for maximum temperature, minimum temperature and relative humidity,

$$E_{i,t} = \sum_{d=t-89}^{t} e_{i,d} \qquad\text{(4)}$$

which is evapotranspiration summed over 90 days (mm).

Rainfall is averaged over a longer window than the other variables because infiltration reaches the water table with a delay. Temperature and humidity describe conditions, so they are averaged, and evapotranspiration is an accumulated loss, so it is summed. The weather series begins on 1 January 2013, so readings from January 2013 have shorter windows: 51 readings have fewer than 90 days of history and 88 have fewer than 180 days of rainfall history. Later readings are not affected. For the evapotranspiration sum a shortened window underestimates the accumulation.

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

The signs agree with the expected physics: more antecedent rainfall and humidity go with a shallower water table, and higher temperature goes with a deeper one. The negative sign for evapotranspiration is not a causal effect, because POWER evapotranspiration peaks in the wet season and shares the monsoon pattern of rainfall. These are descriptive correlations only, and how much each input contributes to the forecast is for the model to determine.

### Data Limitations

Four limitations affect how the results should be read. First, the groundwater record is sparse in time. Readings are quarterly, wells have between 1 and 43 readings, and few wells have long unbroken runs of consecutive quarters, so the number of usable training sequences is much smaller than the 1,232 readings suggest. The strength of the present network lies in the number of wells and their spatial coverage, not in the volume of data per well. Second, 19 of the 53 wells have no reading after January 2021, so the later period is represented by 34 wells. Third, the coarse temperature grid contributes almost no spatial variation between wells. Fourth, humidity and evapotranspiration are reanalysis values that have not been checked against local ground measurements. We also examined telemetry data from the Tamil Nadu state network and did not use it. Only 2 of the 53 wells have a telemetry station within 1 km, the sign convention of the depth values is not documented, and the record could not be matched to the manual readings at the one pair of stations that could be checked. We leave telemetry for future work.

## Methodology

The methodology has two parts. The first, given here, describes how the integrated table of readings becomes a set of supervised examples with a fixed chronological split, and how the wells are joined into a graph. The model that learns from these examples is described in the part that follows.

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

Readings are assigned to calendar quarters, and 21 well-quarter slots held two readings from the same well (1,190 slots held one reading, and none held three or more). We kept the later reading of each pair. We did not average the two values, because an average is a depth that was never measured, and we did not interpolate. The result is 1,211 readings in 43 distinct quarters, from the first quarter of 2013 to the last quarter of 2024. Five quarters contain no reading from any well (2015 Q3, 2016 Q1, 2020 Q2, 2020 Q3 and 2021 Q2), and we did not establish why.

The two readings of a pair can differ widely. The median absolute difference between the discarded and the kept reading is 0.56 m (mean 1.21 m), and the largest is 9.35 m, at Melvastrachavdi in 2013 Q1, where the kept value fits the well's other readings better. The rule is applied the same way to every pair, and no pair was judged individually.

#### Inputs and forecasting target

Let well $i$ have readings on dates $t_{i,1} < t_{i,2} < \dots$. The input vector of the $k$-th reading holds the five weather summaries of Eqs. (2) to (4), evaluated at $t_{i,k}$, and the depth $d_{i,k}$ measured at that reading:

$$\mathbf{x}_{i,k} = \left[\, R_{i,t_k},\; T_{\max},\; T_{\min},\; H,\; E,\; d_{i,k} \,\right]^{\top} \in \mathbb{R}^{6}, \qquad y_{i,k} = d_{i,k+1} \qquad\text{(5)}$$

The target $y_{i,k}$ is the depth at the next reading of the same well. The weather windows end on the reading date, so no input uses information from after it. The last reading of each well has no next reading and therefore no target. This removes 53 readings and leaves 1,158 samples. Four wells have a single reading and give no sample, which leaves 49 wells.

The forecast horizon is the next reading, not a fixed length of time. Of the 1,158 samples, 839 (72.5%) have the next reading in the following quarter, 181 (15.6%) two quarters later, 86 (7.4%) three quarters later and 52 (4.5%) four or more, with the longest gap at 21 quarters. We therefore describe the task as next-reading forecasting, and the elapsed time is kept as a separate quantity, described below.

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

Fitting on all 1,211 readings instead would have shifted the evapotranspiration mean by 3.2 mm and the humidity mean by 0.36 percentage points. These shifts are small, but they would carry information from the test period into training. Values outside the training range are not clipped. In the validation and test sets, at most 1.8% of any column fell outside the training range (the target depth in the test set), and no weather feature exceeded 1.4%.

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

In the gap-tolerant set, 652 windows (77.6%) have the target in the next quarter, 115 (13.7%) two quarters later, 53 (6.3%) three quarters later and 20 (2.4%) four or more. The eight input readings span a median of 912 days (minimum 590, maximum 2,938), against 639 days for the strict windows. In 42 gap-tolerant windows the first input is the first reading of its well, so its gap is undefined and is left empty.

The strict set is thin in time and place. Its training windows have targets only between May 2018 and January 2020, the five empty quarters break every run of consecutive quarters that crosses them, and no strict window has a Thiruvarur well in training or test.

To give later results a floor, we scored the simplest forecast, which repeats the last observed depth. On the gap-tolerant windows its mean absolute error is 0.888 m on validation and 1.347 m on test (root mean squared error 1.338 m and 1.826 m). A learned model is useful only if it improves on these values.

The three negative depths at Allankottai are kept as they are. They appear as the target of 3 gap-tolerant windows and among the inputs of 10, and a check without them is planned for the robustness experiments.

#### Leakage checks

Four conditions were checked by assertions in the code. Every target date is later than the last input date of its window. Every training target is dated on or before $\tau_1$, and every validation and test target after it. The normalisation statistics, recomputed from the readings dated on or before $\tau_1$, equal the stored ones. No window contains readings of two wells. Running the pipeline again reproduces every output file exactly.

The final dataset is therefore 840 windows, each with an $8 \times 6$ block of standardised inputs, the quarter gaps and a target depth in metres, of which 581 train the model, 147 are used to select it and 112 are held back for the final test (Figure 3).

![](figures/figC_split_and_windows.png)

**Figure 3.** Chronological split and the structure of a window. (a) Number of gap-tolerant windows per target quarter, coloured by split. Dashed lines mark the cut dates, 30 September 2022 and 31 December 2023. Quarters without a bar have no window, and five of them have no reading from any well. (b) One window at Keelvalur. The first eight readings of the well (filled points) are the input and the next reading (open circle) is the target. Grey points are later readings. The numbers along the bottom give the quarters between consecutive readings.

### Graph Construction

The graph records which wells are likely to behave alike, so that the model can use the readings of neighbouring wells when it predicts one well. It is built once, from well coordinates alone. No groundwater level, weather value or target enters it, so it carries no information from the validation or test period, and the same graph serves the training, validation and test windows. Construction has four steps: ordering the wells as nodes, computing the distance between every pair, turning distances into edge weights, and normalising the weights. Table 9 summarises the result.

**Table 9. Summary of the well graph.**

| Quantity | Value |
|---|---|
| Nodes (CGWB wells) | 53 (Thanjavur 26, Nagapattinam 16, Thiruvarur 11) |
| Distinct well pairs | 1,378 |
| Distance between wells, minimum / median / mean / maximum | 0 / 49.1 / 50.4 / 125.3 km (standard deviation 24.8 km) |
| Kernel width $\sigma$ | 24.80 km |
| Neighbours per well with $A_{ij} > 0.5$, mean / median | 6.49 / 6 |
| Neighbours per well with $A_{ij} > 0.1$, mean / median | 17.66 / 18 |
| Share of off-diagonal edge weight between districts | 35.7% |
| Row sum of $\mathbf{A}$, minimum / maximum | 2.93 / 13.16 |
| Row sum of $\hat{\mathbf{A}}$, minimum / maximum | 0.805 / 1.095 |

#### Nodes and distances

Each of the 53 wells in the CGWB record is one node, including the 11 wells that give no window. Nodes are sorted by well identifier and numbered 0 to 52, and this order is stored in a file that every later step reads, so that row $i$ of every matrix refers to the same well throughout. Each well has a single latitude and longitude in the CGWB record.

The distance between two wells is the great-circle distance of Eq. (1), computed with $R$ = 6371.0088 km, the mean radius of the Earth. For wells $i$ and $j$ we write it $d_{ij}$. All pairs together give a symmetric $53 \times 53$ matrix with zeros on the diagonal. As an example, Aduthurai (10.98944°N, 79.50000°E) and Ayyampet (10.88889°N, 79.19167°E) are 35.47 km apart. The closest pair of different locations is Paramasivapuram II and III, 0.155 km apart, and the farthest is Peravurani and Puttur, 125.31 km apart. Two pairs of wells share the same coordinates and are recorded under separate identifiers (Icchankottai and Icchankottai1, Vedaranyam and Vedaranyam1). We kept them as separate nodes at distance zero, because they are separate stations in the CGWB record and we could not tell whether they sample the same water level. Three distances (the closest pair, the farthest pair and the closest pair in different districts) were recomputed with an independent haversine routine, and they agree with the matrix to six decimal places.

#### Edge weights

The weight of the edge between wells $i$ and $j$ is a Gaussian function of their distance,

$$A_{ij} = \exp\!\left(-\frac{d_{ij}^{2}}{\sigma^{2}}\right) \qquad\text{(10)}$$

where $\sigma$ sets how quickly the weight falls with distance. A well has weight 1 with itself, because $d_{ii} = 0$, so the matrix already contains a self-connection and none is added. Every pair of wells keeps a weight and no edge is removed by a threshold. With $\sigma$ = 24.80 km, two wells 0.155 km apart have a weight of 0.99996, wells 3.5 km apart 0.980, 10 km 0.85, 25 km 0.36, 50 km 0.017, and the farthest pair, 125 km apart, about $10^{-11}$. The Peravurani and Madukkur pair (19.12 km) has a weight of 0.552.

We set $\sigma$ equal to the standard deviation of the off-diagonal distances, 24.80 km. The value was fixed from the spread of the well locations before any model was trained, and it is not tuned on validation or test results. Table 10 shows what other plausible values would do to the graph. With $\sigma$ = 10 km a well has on average only 1.09 neighbours with a weight above 0.5, and two wells have no neighbour above 0.1, so the graph breaks into small groups. With $\sigma$ equal to the median distance, 49.1 km, a well has 20.3 neighbours above 0.5 and half of the edge weight crosses district boundaries, which blurs the local differences between wells. The chosen value lies between these cases. Table 10 describes the structure of the graph only and says nothing about forecast accuracy. We will repeat the training with $\sigma$ = 10, 25 and 49 km and report the effect on the scores.

**Table 10. Effect of the kernel width on the structure of the graph.**

| $\sigma$ (km) | Rule | Neighbours with $A_{ij} > 0.5$ (mean) | Neighbours with $A_{ij} > 0.1$ (mean) | Wells with none above 0.1 | Weight between districts |
|---|---|---|---|---|---|
| 10.0 | Fixed value | 1.09 | 3.77 | 2 | 22.1% |
| 24.8 | Standard deviation of distances (used) | 6.49 | 17.66 | 0 | 35.7% |
| 25.0 | Fixed value | 6.53 | 17.85 | 0 | 35.9% |
| 49.1 | Median distance | 20.30 | 42.45 | 0 | 50.5% |

At the chosen width every well has at least one neighbour above 0.1. The wells with the weakest connections are Peravurani (row sum of $\mathbf{A}$ 2.93), Vedaranyam and Vedaranyam1 (3.07 each), Puttur (4.39) and Thagattur (4.39), which are far from most other wells, and the best connected is Saliyamangalam (13.16). Of the total off-diagonal weight of 430.5, 153.8 (35.7%) joins wells in different districts, so the graph does not follow the administrative boundaries.

#### Normalisation

Wells with many close neighbours have large row sums, and a graph convolution that adds up the values of the neighbours would give them larger outputs than isolated wells, only because of where they lie. At the chosen width the row sum is 13.16 for Saliyamangalam and 2.93 for Peravurani, a ratio of about 4.5. We remove this effect by the symmetric normalisation of Kipf and Welling [21]. Let $D_i$ be the row sum of $\mathbf{A}$,

$$D_i = \sum_{j=1}^{53} A_{ij}, \qquad \hat{A}_{ij} = \frac{A_{ij}}{\sqrt{D_i\, D_j}}, \quad\text{that is}\quad \hat{\mathbf{A}} = \mathbf{D}^{-1/2}\,\mathbf{A}\,\mathbf{D}^{-1/2} \qquad\text{(11)}$$

where $\mathbf{D}$ is the diagonal matrix with entries $D_i$. Each weight is divided by the square root of the product of the totals of the two wells it joins. Peravurani then keeps a weight of 0.341 on itself, 0.149 on Madukkur and 0.132 on Pattukottai1, while Saliyamangalam, with many neighbours, spreads its weight more thinly, 0.076 on itself and 0.068 on Vadavur. After normalisation the row sums of $\hat{\mathbf{A}}$ lie between 0.805 and 1.095, and the eigenvalues lie between $-1$ and 1, so stacking graph layers does not make the values grow or shrink. The matrix $\hat{\mathbf{A}}$ is the graph given to the model.

#### Checks

The code asserts that the distance and weight matrices are symmetric, that the diagonal of the distances is 0 and that of the weights is 1, that every weight lies in (0, 1], that no row sum is zero, that $\hat{\mathbf{A}}$ is symmetric (largest asymmetry $3 \times 10^{-17}$), that its eigenvalues lie in $[-1, 1]$, and that $\hat{\mathbf{A}}$ equals Eq. (11) element by element. All checks pass, and running the two scripts again reproduces the same matrices.

### Temporal Encoding

The graph tells the model which wells are close. The temporal part must tell it, for each well, what happened over the last eight readings and in what order. This subsection prepares the input of the Transformer encoder. Each window is turned into a sequence of vectors that carry both the readings and their position in time. Nothing is trained here: the encoding consists of one linear layer, whose weights are learned later, and one fixed table. The steps are given in the order in which they are applied.

#### Why the order has to be supplied

The Transformer encoder uses self-attention [22],

$$\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\right)\mathbf{V} \qquad\text{(12)}$$

which compares every row of the sequence with every other row. The result does not depend on the order of the rows. If the eight readings of a window were shuffled, the output would be the same rows in shuffled order, and the model would not know which reading is the most recent. For groundwater this matters. In one Keelvalur window the depths are 4.01, 2.91, 3.96, 4.51, 2.40, 1.68, 4.23 and 4.51 m, and the next reading is 5.10 m. The value 4.51 m appears at position 3 and again at position 7. At position 3 the depth then falls to 2.40 and 1.68 m, so the water table is rising. At position 7 it is the latest observation and the best guide to the next one. Without position information the two rows look the same. The windows described under Sliding windows are ordered from the oldest reading (position 0) to the latest (position 7), and we give this order to the model explicitly.

#### Embedding of the inputs

Each reading $\mathbf{x}_{i,t} \in \mathbb{R}^{6}$ holds the normalised rainfall, maximum temperature, minimum temperature, humidity, evapotranspiration and groundwater depth. Six values are too few to carry a position code and to feed several attention heads, so we first map each reading to a vector of width $d_h = 64$ with one linear layer,

$$\mathbf{e}_{i,t} = \mathbf{x}_{i,t}\,\mathbf{W}_e + \mathbf{b}_e, \qquad \mathbf{W}_e \in \mathbb{R}^{6 \times 64},\ \mathbf{b}_e \in \mathbb{R}^{64} \qquad\text{(13)}$$

The same $\mathbf{W}_e$ and $\mathbf{b}_e$ are used for all eight positions and for all wells, so the layer has 6 × 64 + 64 = 448 parameters. They are initialised at random and learned together with the rest of the model. For a window $\mathbf{W}_{i,k}$ the eight rows are embedded together, which turns the 8 × 6 matrix into an 8 × 64 matrix.

#### Positional encoding

We use the fixed sinusoidal encoding of Vaswani et al. [22]. For position $p \in \{0, \dots, 7\}$ and column pair $m \in \{0, \dots, 31\}$,

$$\mathrm{PE}(p, 2m) = \sin\!\left(\frac{p}{10000^{2m/d_h}}\right), \qquad \mathrm{PE}(p, 2m+1) = \cos\!\left(\frac{p}{10000^{2m/d_h}}\right) \qquad\text{(14)}$$

so that each position receives a vector of 64 values between $-1$ and 1. The 32 column pairs differ in their angular frequency $1/10000^{2m/d_h}$, which falls from 1 for $m = 0$ to $1.3 \times 10^{-4}$ for $m = 31$. The first pairs change quickly from one position to the next and separate neighbouring positions, the last pairs hardly change within eight positions, so the combination of all 32 pairs is different for every position. Position 0 gives $(0, 1, 0, 1, \dots)$ because $\sin 0 = 0$ and $\cos 0 = 1$. Position 1 starts with $(0.8415, 0.5403, 0.6816, 0.7318, \dots)$.

The table has no parameters, it is computed once, and the same 8 × 64 table is used for every window. It depends only on the position in the window and not on the data. It also has a useful geometry. The Euclidean distance between the vectors of two positions depends only on how far apart they are: 1.472 for neighbouring positions (0 and 1, 3 and 4, or 6 and 7 give the same value), 2.719 for positions two apart, 3.581 for three apart, and 4.180 between the first and the last (Figure 4b). Positions that are close in the window therefore have similar codes, and the codes grow apart as the readings get further apart in the window. Compared with a learned position table, the fixed formula adds no parameters, which matters with 581 training windows, and it works unchanged if the window length is changed from 8 to 4 or 12 in the window-length experiments.

![](figures/fig4_positional_encoding.png)

**Figure 4. Positional encoding for a window of eight readings. (a) The 8 × 64 table, with position 0 (oldest reading) at the top. The left columns change quickly with position, the right columns slowly. (b) Euclidean distance between the vectors of every pair of positions; it depends only on the difference of the positions.**

#### Model input

The encoded window is the sum of the embedded readings and the positional codes,

$$\mathbf{H}^{(0)}_{i,k} = \left(\mathbf{W}_{i,k}\,\mathbf{W}_e + \mathbf{1}\,\mathbf{b}_e^{\top}\right) + \mathbf{PE} \in \mathbb{R}^{8 \times 64} \qquad\text{(15)}$$

where $\mathbf{1}$ is a column of eight ones, so that the bias is added to every row, and $\mathbf{PE}$ is the table of Eq. (14). The sum is formed after the embedding, so that the learned layer cannot mix the position code with the readings, and the width stays at 64, whereas concatenating the code would double it. $\mathbf{H}^{(0)}$ is the input of the first Transformer layer. For a batch of $B$ windows it has shape $B \times 8 \times 64$. Table 11 lists the quantities of this step.

**Table 11. Quantities of the temporal encoding.**

| Quantity | Value |
|---|---|
| Input window | 8 readings × 6 features (840 windows: 581 train, 147 validation, 112 test) |
| Embedding width $d_h$ | 64 |
| Embedding layer | one linear layer, 6 × 64 weights + 64 biases = 448 parameters, shared by all positions and wells |
| Positional encoding | fixed 8 × 64 table, no parameters, range −0.990 to 1.000 |
| Distance between position vectors | 1.472 (adjacent) to 4.180 (positions 0 and 7) |
| Output $\mathbf{H}^{(0)}$ | 8 × 64 per window, B × 8 × 64 per batch |

#### Checks

The code asserts that the windows have shape 840 × 8 × 6 with no missing input value, that the split counts are 581, 147 and 112, that the table has shape 8 × 64 with values in $[-1, 1]$, that position 0 equals $(0, 1, 0, 1, \dots)$ and that all eight rows are different. Three entries of the table were recomputed with an independent sine and cosine routine and agree to ten decimal places. The output of the embedding layer for four training windows was also computed by hand as $\mathbf{W}_{i,k}\mathbf{W}_e + \mathbf{1}\mathbf{b}_e^{\top} + \mathbf{PE}$ and agrees with PyTorch to $2.4 \times 10^{-7}$. All checks pass. They show that the encoding is computed as described in Eqs. (13) to (15). They say nothing about forecast accuracy.

#### Limitation

The positional code counts readings, not time. In the gap-tolerant windows two consecutive readings can be one to many quarters apart. In the Keelvalur window above, the gaps between the readings are 1, 1, 1, 2, 3, 2 and 1 quarters, and the code treats the step of three quarters as the same size as the steps of one quarter. The gaps are stored with each window (see Sliding windows) but are not used by the encoding. Hua et al. [2] add a time-interval bias to the attention for this reason. We keep the standard encoding for the main model and treat a time-interval-aware version as a possible extension.


### Spatial Encoding

The temporal encoding of the previous subsection describes one well at a time. The spatial encoder describes the whole network at the moment of the forecast. For a window of well $i$ ending at time $t$, it produces a vector of 64 values that summarises the state of the well and of its neighbours around $i$, weighted by the graph of Eq. (11). It has three steps: building a feature vector for every well at time $t$, passing these vectors through two graph convolution layers, and taking the output at the target well. Nothing is trained in this step. The layers are defined and checked, and their weights are learned later together with the rest of the model.

#### Node features at the forecast time

The graph convolution needs a feature vector for all 53 wells at the same time $t$, the date of the last input reading of the window. The CGWB readings are not available on that scale. Wells are measured on different dates, some have few readings, and a well that is measured quarterly has no value for most days. We therefore give each well its most recent reading on or before $t$ and tell the model how old that reading is. For well $j$ let $\tau_j(t)$ be the date of its latest reading with $\tau_j(t) \le t$, and let $\mathbf{x}_j(t)$ be the six normalised features of that reading (rainfall, maximum and minimum temperature, humidity, evapotranspiration and groundwater depth, in the order of the windows). The node vector is

$$\mathbf{n}_j(t) = \left[\, \mathbf{x}_j(t),\; \tilde{s}_j(t),\; m_j(t) \,\right] \in \mathbb{R}^{8}, \qquad s_j(t) = \frac{t - \tau_j(t)}{91.25}, \qquad \tilde{s}_j(t) = \frac{s_j(t) - \mu_s}{\sigma_s} \qquad\text{(16)}$$

where $s_j$ is the age of the reading in quarters, $\mu_s$ = 4.660 and $\sigma_s$ = 8.307 are the mean and standard deviation of $s_j$ over the observed nodes of the training windows only (30,421 values from 581 windows), and $m_j(t) = 1$ if well $j$ has any reading on or before $t$ and 0 otherwise. If $m_j(t) = 0$, all eight values are set to 0. The indicator is needed because 0 is also the training mean of every normalised feature, so a zero row would otherwise look like an average well. Only readings dated on or before $t$ are used, so no information from the forecast horizon enters the node features, and the node vectors of the validation and test windows use the same rule and the training statistics. Stacking the 53 vectors in the node order of the graph gives the node matrix $\mathbf{N}_{i,k} \in \mathbb{R}^{53 \times 8}$ of the window.

As an example, take the Keelvalur window whose last input reading is dated 24 August 2016 and whose target is the reading of 4 November 2016 (5.10 m). Keelvalur itself has age 0. Tiruvarur was measured on the same date, and Puduchataram 1.12 quarters earlier. Nagapattinam1 has a single reading, taken 14.59 quarters before, and its row is $\tilde{s} = (14.59 - 4.660)/8.307 = 1.195$ with $m = 1$. Vedaranyam1 has no reading yet, so its row is all zeros. Table 12 gives the same information for the whole data set.

**Table 12. Node features of the 840 windows (53 nodes each) by split.**

| Split | Windows | Observed nodes per window (min / median / max) | Age of carried readings, quarters (median / max) |
|---|---|---|---|
| Train | 581 | 52 / 52 / 53 | 0.00 / 37.65 |
| Validation | 147 | 52 / 53 / 53 | 0.00 / 42.66 |
| Test | 112 | 53 / 53 / 53 | 0.00 / 46.67 |

Of the 44,520 pairs of a window and a node (840 × 53), 373 (0.8%) have no reading yet. The median age is zero because most CGWB wells are measured on the same dates, but the distribution has a long tail: some wells are carried forward for up to 47 quarters. Ten of the eleven wells that give no window of their own appear as observed nodes in all 840 windows, and the eleventh, Vedaranyam1, in 467 (55.6%), so they still pass information to their neighbours. In every window the target well has age 0, and its six features equal the last position of the window, which was verified with a largest difference of $1.7 \times 10^{-7}$ (single-precision storage).

#### Graph convolution

Let $\hat{\mathbf{A}} \in \mathbb{R}^{53 \times 53}$ be the normalised adjacency of Eq. (11). Two graph convolution layers of Kipf and Welling [21] give

$$\mathbf{Z}^{(1)} = \operatorname{ReLU}\!\left(\hat{\mathbf{A}}\,\mathbf{N}\,\mathbf{W}_1 + \mathbf{1}\,\mathbf{b}_1^{\top}\right), \qquad \mathbf{Z}^{(2)} = \operatorname{ReLU}\!\left(\hat{\mathbf{A}}\,\mathbf{Z}^{(1)}\,\mathbf{W}_2 + \mathbf{1}\,\mathbf{b}_2^{\top}\right) \qquad\text{(17)}$$

with $\mathbf{W}_1 \in \mathbb{R}^{8 \times 64}$, $\mathbf{W}_2 \in \mathbb{R}^{64 \times 64}$, $\mathbf{b}_1, \mathbf{b}_2 \in \mathbb{R}^{64}$ and a dropout of 0.2 on $\mathbf{Z}^{(1)}$ during training. Each layer first transforms the features of every node with a weight matrix shared by all nodes, then replaces the transformed vector of each well by the weighted sum of the vectors of the wells around it, with the weights taken from the row of $\hat{\mathbf{A}}$, and finally adds a bias and applies the ReLU. In the notation of a single well, $\mathbf{z}^{(1)}_i = \operatorname{ReLU}\big(\sum_{j} \hat{A}_{ij}\,\mathbf{n}_j \mathbf{W}_1 + \mathbf{b}_1\big)$. The sum runs over all wells because we kept every weight and used no threshold, but the weights of distant wells are close to zero. The adjacency already has a weight on its diagonal (Eq. 10 gives $A_{ii} = 1$), so we add no further identity. The bias is added after the multiplication with $\hat{\mathbf{A}}$, and $\mathbf{W}_1$, $\mathbf{W}_2$ are initialised with the Glorot scheme. The encoder has $8 \times 64 + 64 + 64 \times 64 + 64 = 4{,}736$ parameters. The adjacency is a fixed matrix loaded from the file of the graph construction and is not trained. Two layers let a well receive information from wells up to two steps away in the graph. We kept two layers because deeper graph networks tend to make the outputs of neighbouring nodes more alike [21]; we did not test a third layer.

#### Representation of the target well

The graph is applied to all 53 nodes together, and the output for the target well $i$ is its row,

$$\mathbf{z}_{i,k} = \mathbf{Z}^{(2)}_{i,\,:} \in \mathbb{R}^{64} \qquad\text{(18)}$$

This vector is passed on to the fusion with the temporal branch.

#### Reach of the two layers

After two layers the output of well $i$ is built from the rows of $\hat{\mathbf{A}}^2$. Its diagonal entry, the weight of the well's own features, ranges from 0.045 (Valangiman) to 0.237 (Vedaranyam1), with a median of 0.061. Five wells have the smallest self weights (Valangiman, Kodavasal, Needamangalam, Papanasam and Mannargudi, 0.045 to 0.047) and the five with the largest are Vedaranyam1, Vedaranyam, Peravurani, Thagattur and Thillaivilagam (0.136 to 0.237). Four of these five are among the five least connected wells of the graph identified under Edge weights, and Thillaivilagam is the exception. For all 53 wells, more than half of the weight of the row lies on other wells. The graph branch therefore describes the neighbourhood more than the well itself. This is intended: the history of the target well is modelled by the Transformer branch, and the two are combined after both have been computed.

#### Checks

The checks cover the node features and the encoder (Table 13). The age of every carried reading was recomputed with an independent brute-force search for all 44,520 window and node pairs and no difference was found, and the latest date used never exceeds the date $t$ of the window. For the encoder, two layers were recomputed in plain numpy from the weights of the module, and the largest difference is $2.5 \times 10^{-7}$. The outputs on the CPU and on the GPU (NVIDIA RTX 3050, 6 GB) agree to $1.8 \times 10^{-7}$. Dropout changes the output in training mode (largest difference 0.29) and not in evaluation mode. As a test that the graph is used as intended, we added 1.0 to one input feature of a single well in the Keelvalur window and compared the change in the output of Keelvalur. The nearest neighbour in $\hat{\mathbf{A}}$, Nagapattinam1 ($\hat{A}$ = 0.098), changed the output by 0.042, and the farthest well, Puduchataram ($\hat{A}$ = $8.2 \times 10^{-7}$), by $1.1 \times 10^{-4}$, about 400 times less. These tests use untrained random weights, so they show that the code does what Eqs. (16) to (18) say. They say nothing about forecast accuracy.

**Table 13. Checks on the spatial encoder (untrained weights).**

| Check | Result |
|---|---|
| Node matrix shape | 840 × 53 × 8, no missing value |
| Readings used with date later than $t$ | none (largest difference 0 days) |
| Ages and readings against a brute-force search | 0 differences in 44,520 pairs |
| Own features of the target well against the window | largest difference $1.7 \times 10^{-7}$ |
| Encoder output shape | 53 × 64 per window, 64 for the target well |
| Parameters | 4,736 |
| Module against a plain numpy calculation | largest difference $2.5 \times 10^{-7}$ |
| CPU against GPU | largest difference $1.8 \times 10^{-7}$ |
| Effect of +1.0 on a neighbour, nearest / farthest | 0.042 / $1.1 \times 10^{-4}$ |

#### Limitations

Carrying a reading forward means that the node vector of a well can describe a date long before $t$, up to 47 quarters, and the weather features of that reading are equally old. The age feature lets the model discount such rows, but it cannot restore information that was not measured. Wells that have only a single reading, such as Nagapattinam1, contribute that one value to every later window. We chose this design before any model was trained, and we will test it against node features that contain only the well coordinates in the ablation experiments. A version with weather recomputed at time $t$ for every well is a possible extension that we have not built.


## References

[1] The News Minute, "Tamil Nadu's paddy paradox: Cauvery delta turns on the taps even as wells run dry," Mar. 4, 2025. [Online]. Available: https://www.thenewsminute.com/tamil-nadu/tamil-nadus-paddy-paradox-cauvery-delta-turns-on-the-taps-even-as-wells-run-dry

[2] J. Hua, X. Hu, W. Ge, Y. Shen, and C. Kang, "Graph-embedded time-interval-aware transformer for event-driven groundwater level forecasting," *AIP Adv.*, vol. 15, Art. no. 105226, 2025.

[3] A. Thakur, A. S. Chandel, and V. Shankar, "Prediction of groundwater levels using a long short-term memory (LSTM) technique," *J. Hydroinformatics*, vol. 27, no. 1, p. 51, 2025.

[4] A. Wunsch, T. Liesch, and S. Broda, "Groundwater level forecasting with artificial neural networks: A comparison of long short-term memory (LSTM), convolutional neural networks (CNNs), and non-linear autoregressive networks with exogenous input (NARX)," *Hydrol. Earth Syst. Sci.*, vol. 25, pp. 1671-1687, 2021.

[5] C. Zhuang, L. Cui, and Y. Cui, "Enhancing groundwater level prediction with a hybrid deep learning model in Jinan City, China," *Sci. Rep.*, vol. 15, Art. no. 44535, 2025.

[6] L. Wang et al., "A groundwater level spatiotemporal prediction model based on graph convolutional networks with a long short-term memory," *J. Hydroinformatics*, vol. 26, no. 11, pp. 2962-2979, 2024.

[7] Y. Wu et al., "Forecasting groundwater level by characterizing multiple spatial dependencies of environmental factors using graph-based deep learning," *JGR Mach. Learn. Comput.*, vol. 2, Art. no. e2024JH000520, 2025.

[8] M. L. Taccari, H. Wang, J. Nuttall, X. Chen, and P. K. Jimack, "Spatial-temporal graph neural networks for groundwater data," *Sci. Rep.*, vol. 14, Art. no. 24564, 2024.

[9] Q. Zhu, Y. Zhu, J. Niu et al., "Benchmarking machine learning and deep learning models for groundwater level prediction in karst aquifers," *Water*, vol. 18, no. 8, p. 939, 2026.

[10] Y. Liu et al., "Groundwater level prediction using a hybrid TCN-Transformer-LSTM model and multi-source data fusion: A case study of the Kuitun River Basin, Xinjiang," *Sustainability*, vol. 17, no. 19, Art. no. 8544, 2025.

[11] V. H. S. R. Rodrigues et al., "Wavelet gated multiformer for groundwater time series forecasting," *Sci. Rep.*, vol. 13, Art. no. 12726, 2023.

[12] S. Dey and V. Singh, "Forecasting of groundwater level variation under changing climate in Chhattisgarh state using deep learning technique," *Hydrol. Res.*, vol. 56, no. 3, pp. 279-296, 2025.

[13] S. Sankaran, R. Krishnan, and S. Chidambaram, "Assessing and forecasting groundwater resources in the context of climate change using AI techniques for the industry zones in Tiruppur, India," *World*, vol. 7, no. 5, p. 79, 2026.

[14] C. Nolte et al., "Multi-site deep learning for groundwater level prediction across global datasets," *J. Hydroinformatics*, vol. 27, no. 10, p. 1632, 2025.

[15] D. S. Pai, M. Rajeevan, O. P. Sreejith, B. Mukhopadhyay, and N. S. Satbha, "Development of a new high spatial resolution (0.25° × 0.25°) long period (1901-2010) daily gridded rainfall data set over India and its comparison with existing data sets over the region," *Mausam*, vol. 65, no. 1, pp. 1-18, 2014.

[16] A. K. Srivastava, M. Rajeevan, and S. R. Kshirsagar, "Development of a high resolution daily gridded temperature data set (1969-2005) for the Indian region," *Atmos. Sci. Lett.*, vol. 10, no. 4, pp. 249-254, 2009.

[17] Sundaram and Radhakrishnan, "Evaluating the accuracy and applicability of precipitation datasets for sustainable agro-hydrological and climatic planning in Tamil Nadu's Cauvery Delta," *J. Earth Syst. Sci.*, 2025, doi: 10.1007/s12040-025-02690-3. [CONFIRM INITIALS, VOLUME, ARTICLE NUMBER]

[18] A. Saravanan, D. Karthe, S. Ramalingam, and N. Schütze, "Evaluation of remote-sensing- and reanalysis-based precipitation products for agro-hydrological studies in the semi-arid tropics of Tamil Nadu," *Hydrol. Earth Syst. Sci.*, vol. 29, no. 19, pp. 4847-4870, 2025, doi: 10.5194/hess-29-4847-2025.

[19] NASA Langley Research Center, "Prediction of Worldwide Energy Resources (POWER) Project." [Online]. Available: https://power.larc.nasa.gov [CONFIRM PREFERRED CITATION AND ACCESS DATE]

[20] "Drought monitoring in Tamil Nadu using the SPEI index with NASA POWER data, 1991-2024," *Plant Sci. Today*. [Online]. Available: https://horizonepublishing.com/journals/index.php/PST/article/view/4653 [CONFIRM AUTHORS, TITLE, YEAR]

[21] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2017. [CONFIRM AGAINST ORIGINAL]

[22] A. Vaswani et al., "Attention is all you need," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 5998-6008.
