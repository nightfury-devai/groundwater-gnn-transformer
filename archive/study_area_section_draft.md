## Study Area and Data

### Study Area

The study area covers three districts of Tamil Nadu, Thanjavur, Thiruvarur, and Nagapattinam, which together form the core of the Cauvery delta. The 53 observation wells used here lie between 10.26° and 11.29° N and between 78.95° and 79.86° E (Figure 1). The median distance from a well to its nearest neighbour is 6.7 km (mean 7.0 km, maximum 19.1 km), and the largest separation between any two wells is 125.3 km. Thanjavur contributes 26 wells, Nagapattinam 16, and Thiruvarur 11. Of the 1,232 water-level readings, 1,098 are tagged by CGWB to the Cauvery basin (Cauvery Lower sub-basin) and the remaining 134 to the East Flowing Rivers Between Pennar and Kanyakumari basin.

The three districts sit on one continuous alluvial aquifer of sand, silt, and clay laid down by the Cauvery and its distributaries. Because the water-bearing material is a single connected body rather than a set of isolated fractures, as in the hard-rock terrain found elsewhere in Tamil Nadu, the water level at one well carries information about its neighbours. This is the physical basis for representing the wells as a graph.

The districts differ sharply in their groundwater condition. Thanjavur is classified as over-exploited and Thiruvarur as semi-critical, with extraction at 70 to 90% of annual replenishment, while Nagapattinam is left out of the standard CGWB district assessment because all of its assessed groundwater units are saline [1]. The Nagapattinam wells are retained in the network because they record the same aquifer system and the region's salinity problem, but readers should note that their depths reflect a coastal setting where seawater influence is a competing process to abstraction. In 2020 Nagapattinam was divided into Nagapattinam and Mayiladuthurai districts. The well records used here carry the district names assigned in the CGWB files, and no re-assignment of wells between the two successor districts was attempted.

[Figure 1. Study area and the 53 observation wells in Thanjavur, Thiruvarur, and Nagapattinam districts. Source file: figures/full_study_area_map.png]

### Data Sources

Four datasets from three agencies were combined (Table 1). Groundwater depth is the forecasting target. Rainfall, temperature, humidity, and evapotranspiration are the meteorological drivers. All sources cover January 2013 onward. Groundwater readings end on 30 November 2024, and the weather records extend to 31 December 2025.

**Table 1. Datasets used in this study.**

| Variable | Provider | Spatial form | Records | Period |
|---|---|---|---|---|
| Depth to water level (m below ground) | CGWB, via the National Water Data Portal | 53 wells | 1,232 readings | 2013-01-01 to 2024-11-30 |
| Daily rainfall (mm) | IMD | 0.25° grid, 18 cells | 85,464 cell-days | 2013-01-01 to 2025-12-31 |
| Daily maximum and minimum temperature (°C) | IMD | 1° grid, 9 cells | 42,732 cell-days | 2013-01-01 to 2025-12-31 |
| Daily relative humidity (%) and evapotranspiration (mm/day) | NASA POWER | Point value at each well | 251,644 well-days | 2013-01-01 to 2025-12-31 |

#### Groundwater levels

The target is depth to water level in metres below ground level, so a larger value means a deeper water table. The readings come from the CGWB quarterly manual measurement programme. The first 838 readings (to January 2021) came from CGWB records for the 53 wells. A further 394 readings, dated 10 January 2021 to 30 November 2024 and covering 34 of the 53 wells, were added from the CGWB Tamil Nadu quarterly files on the National Water Data Portal. Well names were matched exactly; wells whose names did not appear in the newer file under the same name were not matched approximately, so 19 wells have no readings after January 2021. Re-deriving the earlier period from the new pipeline reproduced all 838 original merged rows without a single difference. The older records are dated one day before the portal's dates; after shifting by one day, 789 of 790 overlapping values were identical, and since readings are later assigned to calendar quarters, this offset matters for only one reading (Vallam, 1 January 2023).

The record is uneven across wells. The number of readings per well ranges from 1 to 43 with a median of 26; 42 wells have at least 9 readings and 33 have at least 17. By district the readings are 552 (Thanjavur), 471 (Nagapattinam), and 209 (Thiruvarur). The number of readings per calendar year varies from 61 (2020) to 142 (2013). Depth to water ranges from -0.90 to 21.32 m, with a median of 3.66 m and a mean of 4.67 m (standard deviation 4.03 m). Mean depth is 5.43 m in Nagapattinam, 4.58 m in Thanjavur, and 3.16 m in Thiruvarur (Table 2). Three readings from a single well (Allankottai) are negative (-0.74 m on 10 January 2021, -0.77 m on 20 January 2023, and -0.90 m on 30 November 2024). They are retained in the dataset; their cause was not established from the records. Of the 1,232 readings, 1,211 fall in distinct well-quarter slots, and 21 are second readings from the same well in the same quarter, which are resolved in preprocessing.

**Table 2. Depth to water level by district (m below ground level).**

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

d = 2R · arcsin( √( sin²(Δφ/2) + cos φ₁ · cos φ₂ · sin²(Δλ/2) ) ),

where φ is latitude, λ is longitude, and R = 6371 km. Matching rainfall and temperature to wells independently was necessary: an earlier attempt that joined them by exact coordinates left 94% of the temperature values empty. The median well-to-cell distance is 11.1 km for rainfall (mean 10.5 km, maximum 20.7 km), and 14 of the 18 rainfall cells are the nearest cell of at least one well. For temperature the median distance is 44.4 km (mean 43.0 km, maximum 66.0 km), and only 3 of the 9 cells are used. Temperature is therefore nearly uniform across neighbouring wells in this dataset, and the spatial differences between wells that the graph can exploit come mainly from the groundwater record itself and from rainfall, not from temperature.

**Temporal aggregation.** A single day's weather does not explain a water level measured on that day, so each reading is paired with trailing-window summaries of the weather before it. For well i with a reading on day t,

R(i,t) = (1/180) · Σ r(i,d), summed over d = t−179, …, t (mean daily rainfall over 180 days, mm day⁻¹),

Tmax(i,t) = (1/90) · Σ τmax(i,d), and likewise Tmin(i,t) and humidity H(i,t), summed over d = t−89, …, t,

E(i,t) = Σ e(i,d), summed over d = t−89, …, t (evapotranspiration summed over 90 days, mm).

Rainfall is averaged over a longer window than the other variables because infiltration reaches the water table with a delay. Temperature and humidity describe conditions, so they are averaged; evapotranspiration is an accumulated loss, so it is summed. Windows are computed from the days available. Because the weather series begins on 1 January 2013, readings taken in January 2013 have shorter windows: 51 readings have fewer than 90 days of history and 88 have fewer than 180 days of rainfall history. No reading after January 2013 is affected. For the 90-day evapotranspiration sum the shortened windows give partial sums and therefore underestimate the true accumulation for those readings.

The integrated table has 1,232 rows and no missing values. Table 3 gives the summary statistics of the model inputs, and Table 4 gives the correlation of each input with groundwater depth.

**Table 3. Summary statistics of the integrated dataset (n = 1,232).**

| Variable | Mean | Std. dev. | Min | Max |
|---|---|---|---|---|
| Rainfall, 180-day mean (mm day⁻¹) | 3.20 | 2.23 | 0.00 | 10.33 |
| Maximum temperature, 90-day mean (°C) | 33.60 | 2.20 | 30.01 | 37.68 |
| Minimum temperature, 90-day mean (°C) | 24.95 | 1.33 | 21.91 | 27.00 |
| Relative humidity, 90-day mean (%) | 73.58 | 8.23 | 52.08 | 88.02 |
| Evapotranspiration, 90-day sum (mm) | 56.53 | 47.19 | 0.02 | 159.38 |
| Depth to water level (m) | 4.67 | 4.03 | -0.90 | 21.32 |

**Table 4. Correlation of each input with depth to water level.**

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
