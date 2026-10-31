# Spatio-Temporal GNN-Transformer Framework for Groundwater Level Forecasting in the Cauvery Delta Alluvial Aquifer, India

## Abstract

[ABSTRACT, TO BE WRITTEN LAST, AFTER ALL RESULTS EXIST]

## Introduction

Groundwater is the primary water source for large parts of the Cauvery delta [2], and the pressure on it is not hypothetical. In Thanjavur district, CGWB's official assessment already classifies groundwater extraction as over-exploited; Thiruvarur is classified as semi-critical, at 70–90% of its annually replenished supply. Nagapattinam does not appear on the same scale at all. CGWB excludes it from its standard district assessment entirely, because every one of its assessed groundwater units is saline and unusable, a finding reported by hydrologist Dr. S. Janakarajan of SaciWATERs in a March 2025 investigation of CGWB data by The News Minute [1]. Two mechanisms drive this: agricultural over-extraction lowering the water table, and seawater intrusion from the adjoining Bay of Bengal moving in to replace it. Both operate across the same physically connected aquifer, which is precisely why a model built around independent, per-well time series is ill-suited for this problem.

Thanjavur, Thiruvarur, and Nagapattinam, the three districts making up the study area, sit on a single continuous alluvial aquifer of sand, silt, and clay deposited by the Cauvery over a long geological history. Wells drilled into this aquifer are not isolated observation points; they draw on a hydraulically connected body of water, so a change measured at one well carries real information about what is likely happening at its neighbours. Forecasting groundwater levels here without representing that connectivity throws away the spatial structure that makes the problem tractable at a network scale, rather than one well at a time.

Existing research on this problem has so far only addressed parts of it, not all of it at once. As the following section details at length, per-well temporal modelling, an LSTM or a similar recurrent architecture applied independently to each well, remains the dominant approach in Indian groundwater studies, with no mechanism for one well's dynamics to inform another's. Where researchers have built an explicit graph over a well network, the temporal component paired with it has almost always been a recurrent or convolutional architecture rather than a Transformer, even in the most spatially explicit of these efforts. The converse gap also exists: studies using genuine Transformer-style attention for the temporal side typically model each well independently, with no spatial mechanism at all. Only one study identified in this survey, Hua et al. [2] combines a real spatial graph with a real Transformer encoder, applied to eight wells in California, using groundwater-level records alone, with no rainfall, temperature, or other weather variable entering the model anywhere.

This work applies a hybrid graph neural network and Transformer architecture to the Cauvery delta's alluvial aquifer, fusing CGWB groundwater records with IMD weather data to capture the monsoon-driven recharge and depletion cycles that govern this system. To the best of our knowledge, this is the first application of spatio-temporal GNN-Transformer modelling to alluvial aquifer dynamics in the Cauvery delta specifically, fusing CGWB-IMD data to capture monsoon-recharge-driven groundwater depletion in a physically connected rural aquifer system.

This work makes the following contributions:

1. A hybrid architecture combining explicit graph-based spatial modelling of well-to-well hydraulic connectivity with a Transformer-based temporal encoder, rather than treating either side as an afterthought to the other.
2. Application of this architecture to a 53-well network, fusing CGWB groundwater records with IMD rainfall and temperature data, a substantially larger well network than the only existing precedent combining a spatial graph with a Transformer (8 wells, Hua et al. [2]), offering the model more genuine well-to-well relationships to learn from than networks an order of magnitude smaller.
3. Application to a physically connected alluvial delta aquifer, where the underlying hydrogeology gives the spatial graph a real physical basis rather than an assumed one.

## Related Work

Groundwater forecasting in India has, with rare exception, treated each well as its own closed problem, or treated an entire district as a single pooled signal with no wells at all. Thakur et al. [3] applied a hybrid PCA-DWT-LSTM pipeline to nine wells in Himachal Pradesh, drawing on CGWB-style records fused with CHIRPS rainfall, FLDAS evapotranspiration and soil moisture, and TerraClimate temperature, a data-source mix close in spirit to the one used here. Every well in that study was modelled with its own independent LSTM; nothing in the pipeline allows information from one well to inform the forecast at another, even where wells sit within the same aquifer system. Dey and Singh [12] went simpler still, forecasting groundwater levels across 267 wells in Chhattisgarh with a two-layer fully connected network, no recurrent unit, no spatial term, and no attention mechanism of any kind, the plainest architecture among the thirteen papers surveyed here. Their own discussion names attention-based deep learning as a direction they did not pursue. Sankaran et al. [13] worked in Tiruppur, Tamil Nadu, the same state as the present study and the geographically closest of the thirteen papers reviewed, but rather than modelling wells individually or as a graph, they pooled multiple monitoring wells into a single regional composite signal before applying an ICEEMDAN-VMD decomposition feeding an SMA-optimised LSTM and a CNN-LSTM ensemble. Pooling wells this way erases whatever spatial structure the underlying network might have carried, and the authors say as much themselves, flagging both well-specific spatial modelling and Transformer-based architectures as promising directions they left unexplored. Nolte et al. [14] took the opposite scaling choice, training a single global model on more than 1,800 wells across nine regions, including 200 in India, but represented each well only through a trainable embedding vector rather than any message-passing structure, paired with a plain single-layer LSTM; the 200 Indian wells are absorbed into a much larger pooled dataset with no CGWB-specific detail, and the authors state outright that benchmarking against GNN or Transformer architectures remains an open research question. This is not a limitation specific to Indian groundwater work. Wunsch et al. [4] compared NARX networks, LSTM, and 1D-CNN across seventeen wells in the Upper Rhine Graben and found NARX generally the strongest performer, but the comparison itself, like almost every per-well benchmark before it, was run well by well, with no channel through which one observation well's state could shape the prediction at a neighbouring one. Read together, these studies illustrate a broader habit in the field: recurrent, convolutional, or plain feedforward temporal models are compared and refined in detail, while the spatial arrangement of the wells producing the data is either ignored outright or collapsed into a single pooled signal.

A smaller set of studies has tried to close that gap by building an explicit graph over the well network, but the temporal component paired with the graph has consistently stopped short of a full attention-based encoder. Zhuang et al. [5] connected twenty-seven wells in Jinan, China, through a K-nearest-neighbour graph processed with GraphSAGE, feeding the resulting spatial embeddings into a multi-branch GRU that operates at three separate temporal scales before a cross-attention layer merges the branches with the graph output. That cross-attention step blends the model's own temporal branches together; it does not replace the GRU as the sequence encoder, so the architecture is closer to graph-plus-recurrent-plus-fusion-attention than to a genuine GNN-Transformer pairing. Wang et al. [6] took a related route in Xiangtan, building a Delaunay-triangulated graph over sixteen wells with edge weights combining distance and terrain similarity, then passing the graph output through an LSTM with no attention mechanism anywhere in the pipeline. Wu et al. [7] went further on the spatial side, fusing five separate adjacency matrices, built from distance, rainfall correlation, hydraulic gradient, water-level correlation, and sub-basin membership, adaptively weighted during training across fifteen wells on the North China Plain, yet the temporal half of that model remained a GRU with only a shallow attention layer added on top of its hidden states, not a stacked self-attention encoder of the kind a Transformer provides. Taccari et al. [8] built a large-scale spatial-temporal graph network across 213 piezometers in the Dutch Overbetuwe polder, connecting each well to its nearest neighbours and nearest exogenous stations, and paired that graph with dilated gated convolutions rather than any recurrent or attention-based unit; the model outperformed a MODFLOW numerical simulation on the majority of test piezometers, but the temporal mechanism is convolutional throughout, and the setting, a flat, non-monsoonal polder governed by river stage and pumping rather than seasonal recharge, has little in common with an alluvial delta aquifer.

The reverse pattern appears just as often: a genuine Transformer or attention-based temporal encoder, with no spatial structure connecting the wells at all. Zhu et al. [9] benchmarked nine architectures, including a four-head Transformer, across three karst monitoring sites in Guilin, deliberately keeping the sites independent so that differences in forecasting accuracy could be attributed to hydrogeological complexity rather than to any spatial-modelling choice; no graph or cross-well mechanism was ever part of the design. Liu et al. [10] chained a temporal convolutional network into a Transformer encoder and then an LSTM for three wells in the Kuitun basin in Xinjiang, and while the middle stage genuinely applies multi-head self-attention, each of the three wells is modelled on its own, with no spatial term linking them. Rodrigues et al. [11] built two parallel Transformer-family encoders, a standard Transformer and a wavelet-based Crossformer variant, merged through a learned gate, to forecast groundwater levels from eight wells and twenty-six other sensors in Brazil's Urucuia aquifer; a separate statistical cross-correlation analysis was run afterward to describe how the wells relate to each other, but that analysis sits outside the forecasting architecture itself and never feeds back into it as a graph or adjacency structure.

Only one study identified across this survey combines a real spatial graph with a real Transformer-based temporal encoder in the way this work does. Hua et al. [2] built a distance-weighted graph attention network over eight wells in California and paired it with a full Transformer encoder carrying a time-interval-aware attention bias, developed specifically to handle the irregular sampling and abrupt recharge events in that dataset. It stands, of the thirteen papers surveyed here, as the only prior instance of the GNN-Transformer combination applied to groundwater forecasting anywhere, and it does so without incorporating rainfall, temperature, or any other weather variable, on a well network an order of magnitude smaller than the one considered in this study.

## Study Area and Data

### Study Area

The study area covers three districts of Tamil Nadu, Thanjavur, Thiruvarur, and Nagapattinam, which together form the core of the Cauvery delta. The 53 observation wells used here lie between 10.26° and 11.29° N and between 78.95° and 79.86° E (Figure 1). The median distance from a well to its nearest neighbour is 6.7 km (mean 7.0 km, maximum 19.1 km), and the largest separation between any two wells is 125.3 km. Thanjavur contributes 26 wells, Nagapattinam 16, and Thiruvarur 11. Of the 1,232 water-level readings, 1,098 are tagged by CGWB to the Cauvery basin (Cauvery Lower sub-basin) and the remaining 134 to the East Flowing Rivers Between Pennar and Kanyakumari basin.

The three districts sit on one continuous alluvial aquifer of sand, silt, and clay laid down by the Cauvery and its distributaries. Because the water-bearing material is a single connected body rather than a set of isolated fractures, as in the hard-rock terrain found elsewhere in Tamil Nadu, the water level at one well carries information about its neighbours. This is the physical basis for representing the wells as a graph.

The districts differ sharply in their groundwater condition. Thanjavur is classified as over-exploited and Thiruvarur as semi-critical, with extraction at 70 to 90% of annual replenishment, while Nagapattinam is left out of the standard CGWB district assessment because all of its assessed groundwater units are saline [1]. The Nagapattinam wells are retained in the network because they record the same aquifer system and the region's salinity problem, but readers should note that their depths reflect a coastal setting where seawater influence is a competing process to abstraction. In 2020 Nagapattinam was divided into Nagapattinam and Mayiladuthurai districts. The well records used here carry the district names assigned in the CGWB files, and no re-assignment of wells between the two successor districts was attempted.

[Figure 1 (figures/fig1_study_area.png). Study area and the 53 CGWB observation wells, by district: Thanjavur (n = 26), Thiruvarur (n = 11), and Nagapattinam (n = 16). District outlines are taken from the boundary file used for the map, and the Nagapattinam outline includes the area that became Mayiladuthurai district in 2020. A few wells lie on or very close to a district line, so their colour may not match the polygon they sit in.]

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

where φ is latitude, λ is longitude, and R = 6371 km. The matching is shown in Figure 2. Matching rainfall and temperature to wells independently was necessary: an earlier attempt that joined them by exact coordinates left 94% of the temperature values empty. The median well-to-cell distance is 11.1 km for rainfall (mean 10.5 km, maximum 20.7 km), and 14 of the 18 rainfall cells are the nearest cell of at least one well. For temperature the median distance is 44.4 km (mean 43.0 km, maximum 66.0 km), and only 3 of the 9 cells are used. Temperature is therefore nearly uniform across neighbouring wells in this dataset, and the spatial differences between wells that the graph can exploit come mainly from the groundwater record itself and from rainfall, not from temperature.

[Figure 2 (figures/fig2_data_integration.png). Nearest-cell matching of the 53 wells to (a) the 18 IMD rainfall cells (0.25 degree) and (b) the 9 IMD temperature cells (1 degree). Shaded cells are the nearest cell for at least one well; dashed cells are not used. Lines join each well to the centre of its matched cell. Temperature cells T1, T2, T3, T8, and T9 fall outside the plotted extent.]

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

### References

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
