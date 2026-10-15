# Monitoring deserts: mapping where air-quality monitors are scarce in South Asia, and who lives there

*Tommaso Isolabella. Code, data sources and figures are in the [repository](../README.md); every number below is read from the saved outputs collected in [`key_numbers.yaml`](key_numbers.yaml).*

## Summary

Ground-level fine particulate matter (PM2.5) is monitored by a few hundred stations across South Asia, most of them in cities. I trained a model to predict monthly PM2.5 from satellite and weather data, scored it by holding out whole regions rather than random rows, applied it to every 5 km cell of the study area, and used the result to ask two questions: how far are people from the nearest monitor, and are the gaps concentrated among the poor?

The main results are these. Predicting a region the model has not seen is much harder than ordinary cross-validation suggests ($R^2 = 0.27$ against $0.60$). The satellite and weather features nevertheless carry real information beyond interpolating the stations. A quarter of the people in the study countries live more than 100 km from a monitor, and over half more than 50 km. The poorest fifth live farther from monitors than the richest fifth, but most of that gap is explained by the difference between urban and rural areas, not by wealth itself. The model's uncertainty is large, and the findings are best read as patterns and rankings, not as precise concentrations.

## 1. Why this question

Air pollution is among the largest environmental risks to health, and policy depends on measuring it. Reference monitors are expensive, so networks cluster where pollution is already suspected to be a problem. That leaves "monitoring deserts": places where nobody measures. If the deserts are also where poorer people live, the people with the least voice are also the least measured. A map of the deserts, and a clear account of how far a model can fill them in, is the kind of evidence a public-health group or a sensor-placement programme can use.

## 2. Data

- **Ground truth:** monthly PM2.5 from the OpenAQ network, 343 stations and 5,222 station-months, March 2025 to June 2026.
- **Satellite:** tropospheric nitrogen dioxide and the absorbing aerosol index from Sentinel-5P TROPOMI, monthly means, extracted at stations (Earth Engine) and on the whole grid.
- **Weather:** ERA5 monthly reanalysis: temperature, dew point, relative humidity, wind, boundary-layer height, pressure and precipitation.
- **People and income:** WorldPop 2020 population on a 1 km grid for India, Pakistan, Bangladesh, Nepal and Sri Lanka; Meta's Relative Wealth Index (RWI); night-time light.
- **The grid:** 0.05 degree cells (about 5 km), 434,940 cells over the study region, of which about 155,000 have a population.

## 3. Method

**Features and target.** The target is $\log(1 + \mathrm{PM}_{2.5})$, because concentrations are positive and right-skewed. The thirteen features are NO2, the aerosol index, nine weather variables, and the month as a sine and cosine. Latitude and longitude are deliberately left out so that the model learns the physical relation, not "which station is this".

**Spatial cross-validation.** Random cross-validation lets a held-out station sit a few kilometres from a training station, so the model is rewarded for memorising local conditions. Instead, the 343 stations are clustered into 35 spatial blocks, and the blocks are packed into eight folds so that a whole region is held out together. The number of blocks and folds was chosen by measuring the distance from each held-out station to the nearest training station and comparing it with the distance from a real prediction cell to the nearest station ([methodology note](../docs/methodology/spatial_cv.md)). With random folds over rows, a held-out station's other months are in the training set, so its distance to the training data is zero; the spatial folds give a median of about 139 km, against 91 km for the populated cells to be predicted. Hyperparameters were tuned on the spatial folds too, never on random ones.

**Models and baselines.** A constant (the training mean), the nearest station, inverse-distance weighting of the five nearest stations of the same month, a Ridge regression, a Random Forest and XGBoost. The interpolation baselines use only station locations and values, so they answer the question "do the features add anything beyond interpolating the stations?". Models are compared with paired tests on the per-fold errors, and a gap inside the fold-to-fold noise counts as a tie, in which case the simpler model wins.

**Prediction and uncertainty.** The definitive model predicts $\log(1+\mathrm{PM}_{2.5})$ for every cell and month, back-transformed with $e^x - 1$ and clipped at zero. Three indicators accompany each prediction: a 90% band from the out-of-fold residuals of the spatial cross-validation, the disagreement between the Ridge and the Random Forest, and the number of features that lie outside the range seen at the stations. None is a calibrated interval ([methodology note](../docs/methodology/uncertainty.md)).

## 4. How good is the prediction?

![Baseline ladder](figures/cv_ladder.png)

Pooled over the eight held-out regions, the Ridge reaches $R^2 = 0.27$ (RMSE 27 µg/m³, MAE 16 µg/m³), the Random Forest $0.27$ and XGBoost $0.25$; the constant and the nearest station score below zero, and inverse-distance weighting $0.12$. The Random Forest is clearly better than inverse-distance weighting (paired $t = -3.4$, better in seven of eight folds) and indistinguishable from the Ridge ($t = 0.4$ for the Ridge minus the forest). The same Random Forest scored on random folds reaches $R^2 = 0.60$, more than twice the held-out-region figure, which is the leakage that spatial cross-validation removes.

The Ridge is the definitive model for three reasons: it ties the more flexible models, it is the simplest, and it can extrapolate beyond the training range, whereas a forest can only return averages of values it has seen ([methodology note](../docs/methodology/ridge_regression.md)). Its standardised coefficients are dominated by temperature, dew point and relative humidity, which together describe the seasonal and meteorological state; the aerosol index (0.17) and NO2 (0.07) are small. The skill is therefore mostly seasonal and meteorological, with the satellite columns adding comparatively little.

One region is much harder than the rest. The fold that holds out the Delhi region has a negative per-fold $R^2$ for every model. The predictions there are compressed towards the middle although they rank the station-months well, so any statement about the most polluted places should be about their rank, not their level.

![Uncertainty](figures/uncertainty.png)

The uncertainty band illustrates the limits. It holds 90% of the held-out observations on average, but only 78% in the most polluted fold; a single width cannot adapt to a region the model finds harder. And 48% of predicted cell-months have three or more features outside the stations' range, where a linear model extrapolates.

Three sanity checks that need no new data support using the surface as a pattern. The predicted population-weighted national seasonal cycle follows the observed station mean (correlation 0.98 over 16 months). Nine of ten spot-checked places behave as expected: Delhi, Lahore, Lucknow, Patna, Dhaka and Kolkata are in the top fifth of populated cells and Munnar, Nuwara Eliya and Port Blair in the bottom 5%. Leh, at high altitude, is predicted in the middle of the distribution, which points to the missing elevation feature.

## 5. Where are the deserts, and who lives in them?

For every cell the distance to the nearest station is computed on the sphere, and a cell is in the desert if that distance exceeds a threshold. There is no single correct threshold (how far a station's reading generalises depends on land use and pollutant), so the results are shown from 25 to 200 km. The median populated cell is about 91 km from a station. Of the roughly 1.8 billion people in cells with population data, 77% live more than 25 km from a station, 56% more than 50 km, 27% more than 100 km and 9% more than 200 km. Over the five study countries, 45% of the land area is more than 100 km from a station.

![Share of each income group in the desert](figures/equity_desert_share.png)

People are split into five income groups holding equal numbers of people, by the Relative Wealth Index. The poorest fifth live farther from monitors: beyond 25 km live 92% of them against 43% of the richest fifth, a gap of 49 percentage points (95% interval 31 to 62). The gap narrows with distance (16 points at 100 km, interval 6 to 24) and is indistinguishable from zero at 200 km.

![Gap and correlations](figures/equity_gap.png)

Most of this is an urban-versus-rural gap. Monitors are in dense, lit cities, and cities are richer than the countryside. The rank correlation between a cell's distance to a station and its RWI is −0.25 (interval −0.30 to −0.18). It shrinks to −0.08 after controlling for population density, and to 0.01 (interval −0.03 to 0.06) after controlling for density and night-time light. A second income proxy, night-time light per person, shows gaps of the same sign at every threshold, but the intervals include zero. The intervals come from a bootstrap over 3 degree blocks of the map, because neighbouring cells are strongly correlated and a cell-level bootstrap would make every result look far more certain than it is.

The predicted pollution in the desert is not clearly different from that outside it: the population-weighted mean prediction in desert cells (more than 100 km from a station) is about 1 µg/m³ above that in monitored cells for the poorest fifth and 0.2 µg/m³ above for the richest, differences far smaller than the model error. Where a new monitor would cover the most exposure (predicted PM2.5 times population) is ranked over 0.5 degree zones of the desert and shown on the [interactive map](../docs/map.html); the ranking is dominated by population, so it is stable when the predictions are moved to the lower or upper bound of the band.

## 6. Limitations

1. **Stations are not placed at random.** They are in towns where pollution is a concern. The training data may not represent the unmonitored areas, and predictions far from any station should be trusted less than the band alone suggests, because the band comes from residuals measured at stations.
2. **The validated range is shorter than the range predicted over.** Cross-validation tests transfer at about 90 to 230 km from the nearest training station, while populated cells have a median of 91 km and a 90th percentile of 375 km. The scores are conservative for cells near a station and say nothing about the most remote cells.
3. **Columns, not the surface.** TROPOMI measures atmospheric columns, and the relation to ground-level concentration depends on boundary-layer physics that the model only approximates through the weather features.
4. **Proxies.** The Relative Wealth Index is a modelled product and night-time light per person is confounded with density; neither is income. The equity result is an association, not a cause, and the Relative Wealth Index is licensed for non-commercial use.
5. **The desert threshold is a judgement.** The results are shown across thresholds, not at one.
6. **Short and uneven period.** Sixteen months, with March to June covered twice and the other months once; everything describes this period. The grid also stops at 93.4 degrees east, and the notebooks do not mask sea and other countries (the report figures do).
7. **Compressed high values and an uneven band** (above).

The full list of open items is in [`docs/known_issues.md`](../docs/known_issues.md).

## 7. How this generalises

The pipeline has no South Asia-specific step. The station census, the Earth Engine and ERA5 requests, the grid, and the spatial cross-validation all take a bounding box and a period from a configuration file. Extending it to another region needs a station network with enough history, population and wealth layers for the countries involved, and a re-check of the distance diagnostic: the block size of the spatial folds should be chosen by matching held-out distances to the distances the predictions will span, and that comparison is a one-cell diagnostic in notebook 08. Extending it to a global scale would mainly be a matter of compute (the grid-feature download is the slow step), and of the heterogeneity of station networks and data quality between regions, which is where the uncertainty indicators would matter most.
