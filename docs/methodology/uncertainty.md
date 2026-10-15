# Uncertainty estimation: the strategies, and where this project stands

The phase names used below (Phase I, Phase L and so on) are the pipeline stages listed in [../pipeline.md](../pipeline.md); section numbers are kept from the original write-up so cross-references between the pages read correctly.


Phase L attaches three indicators to every predicted cell (notebook 11, Step 3): a residual band, the disagreement between the Ridge and the Random Forest, and a count of features outside the range of the stations. This page places them among the other ways of estimating the uncertainty of a prediction, says what each captures and misses, and lays out how to check and improve what is there. It is written for a reader who knows the statistics of measurement error from the physical sciences and needs the machine-learning versions of the same ideas.

### G.1 What "uncertainty" means for a predicted cell

A predicted value can be wrong for several different reasons, and a single number rarely covers all of them. In the order they matter here:

1. **Irreducible scatter.** Two cells with identical features still differ in PM2.5, because of sources and weather the features do not describe. This is the noise around the "true" relationship and does not shrink with more data.
2. **Model error (bias).** The linear form with these thirteen features misses structure that exists: interactions, thresholds, local sources.
3. **Parameter uncertainty.** Different training samples would give slightly different coefficients. With about 5,200 station-months and 13 coefficients this is small compared with the first two, though the station-months are correlated, so it is less small than it looks.
4. **Extrapolation (covariate shift).** The model is used on cells whose feature values, or combinations, the stations never covered. The error there is larger and cannot be estimated from stations that do not resemble those cells.
5. **Input error.** Satellite retrievals and ERA5 values at a cell are themselves uncertain and are treated as exact.
6. **Representativeness.** A station is a point; a grid cell is about 5 km across. The target the model learns is a point value, and the quantity wanted is a cell-scale one.
7. **Spatially correlated error.** The mistakes of neighbouring cells are not independent. This does not change the uncertainty of one cell but changes that of any sum or average over cells, which is what the equity analysis (Phase N) computes.

### G.2 What a good estimate must do

- **Calibration:** a 90% band should contain the true value about 90% of the time, not just on average over everything but across regions, months and pollution levels.
- **Sharpness:** among calibrated bands, narrower is better. A band that is wide everywhere is calibrated and useless.
- **Usefulness for the decision:** the monitor-placement use needs to know whether a cell's *rank* is reliable, and the equity use needs uncertainty on *aggregates*. A pointwise band answers neither directly.
- **Warnings where it matters:** at minimum, flag where the estimate should not be trusted, even if it cannot be quantified.

### G.3 The strategies

**1. Analytic standard errors of the regression.** For ordinary least squares and for ridge there are textbook formulas for the variance of the coefficients and of a prediction. They assume independent errors with constant variance, usually Gaussian. Station-months from the same station or region are correlated and the spread of PM2.5 errors grows with the level, so the formulas understate the uncertainty, and they cover reasons 1 and 3 only. For a penalised fit the standard formulas are not even exact. Not used.

**2. Resampling and ensembles of the same model.**
- *Spread of the trees in a forest.* The standard deviation across the trees' predictions is cheap, and the original proposal for the prediction step. It measures how much the forest varies from tree to tree, which is related to parameter uncertainty (reason 3) and not to scatter (reason 1), so it typically understates the real error and carries no coverage guarantee (Wager, Hastie and Efron, 2014, show what the spread does estimate). Not available for the Ridge.
- *Bootstrap of the Ridge.* Refit on resampled data and look at the spread of the predictions. Resample whole stations or spatial blocks, not rows, or the correlation inside a station makes the spread too small. It captures reason 3 only and costs one refit per replicate, which for a Ridge is cheap.

**2b. Ensembles of different models (used here as indicator (b)).** Where models of different kinds disagree, at least one is wrong. This is a warning that works without any distributional assumption. It does not say which model is right, both may be wrong together, and the size of the disagreement is not an error bar.

**3. Empirical residual band (used here as indicator (a)).** Take the out-of-fold residuals of the spatial cross-validation and use their quantiles as the offsets around any prediction (notebook 11, Step 3). It covers reasons 1 and 2 as they appeared at the stations, and, because the folds are spatial, part of reason 4. Its weakness is that the width is global. It is a special case of the conformal methods below, without their finite-sample correction.
Variants that make it local, in rising effort:
- *By group:* separate offsets per month, per region or per prediction level (binned), from the residuals in that group. Needs enough residuals per group.
- *Scaled residuals:* model the size of the error as a function of features or of an indicator (for example fit a second model to $|\text{residual}|$), divide the residuals by it, take quantiles of the scaled ones, and multiply back for a cell. This is where the disagreement and the range count could be used as the scale.

**4. Conformal prediction.** A family of methods that turn any model plus a set of held-out errors into intervals with a coverage guarantee that makes no assumption about the model or the error distribution, only that the calibration points and the new points are exchangeable (Vovk et al., 2005; Angelopoulos and Bates, 2021 for an introduction).
- *Split conformal* (Lei et al., 2018): the method described in G.3(3), plus a small correction (use the $\lceil (n+1)(1-\alpha) \rceil / n$ quantile instead of the plain quantile). A direct upgrade of what is in place.
- *Cross-conformal / jackknife+* (Barber et al., 2021): uses every point for both fitting and calibration through cross-validation, as the spatial folds already do. Closest to the existing out-of-fold setup.
- *Normalised conformal:* the scaled-residual variant above, which gives bands that vary from cell to cell.
- *Under covariate shift* (Tibshirani et al., 2019): if the cells to be predicted differ from the calibration stations, weight the calibration residuals by how much each station resembles the target cells (a density ratio). It is the theoretically correct answer to reason 4, and in practice the weights are hard to estimate and unstable when the shift is large, so it needs care.
The guarantee is *marginal* (averaged over the calibration and target population), not conditional on a cell, and it relies on exchangeability, which spatially clustered stations only approximately satisfy. Libraries (for example MAPIE; check what is installed in `mon-des`) implement the standard variants, but the spatial folds must be passed in as the cross-validation scheme.

**5. Quantile methods.** Model the quantiles of the target directly. *Linear quantile regression* (Koenker and Bassett, 1978; `sklearn.linear_model.QuantileRegressor`) fits one linear model per quantile, so the band width can depend on the features, and has the same readability as the Ridge. *Gradient boosting with a quantile loss*, and *quantile regression forests* (Meinshausen, 2006), fit flexible quantiles but give up the simplicity that motivated choosing the Ridge. All of them need their coverage checked on held-out regions, since the quantile models can be miscalibrated outside the training data, and they have no automatic extrapolation guarantee.

**6. Probabilistic and Bayesian models.**
- *Bayesian ridge* (`sklearn.linear_model.BayesianRidge`): the same model with a posterior over the coefficients and an estimate of the noise level. The predictive variance combines parameter uncertainty and noise but assumes constant Gaussian noise, so it inherits the weakness of G.3(1).
- *Gaussian-process regression / kriging* (Diggle and Ribeiro, 2007): models the errors as spatially correlated, so the variance grows with distance from the stations and shrinks near them, which is exactly the property the "monitoring desert" question cares about. It also gives the correlation needed for aggregates (reason 7). It needs the errors to have spatial structure: the interpolation baselines of notebook 09 did poorly at the distances involved, which suggests little spatial dependence beyond what the features explain, so the benefit is uncertain (a semivariogram of the model residuals is the test; it has not been done). The cost is also high: an exact fit scales with the cube of the number of stations, which is fine for hundreds of stations, but the prediction over 7 million rows needs approximations.

**7. Applicability and novelty indicators (used here as indicator (c)).** They do not estimate an error, they say whether the model should be used at all for this row.
- *Per-feature range counts* (what is implemented): simple and readable, but marginal. A cell can be inside every range and still combine features in an unprecedented way.
- *Distance in feature space:* the distance from a row to its nearest training rows in the standardised feature space, or the Mahalanobis distance from the training distribution, so joint novelty counts.
- *Dissimilarity index and area of applicability* (Meyer and Pebesma, 2021): a distance to the nearest training row in the feature space weighted by the importance of each feature, with a threshold taken from the cross-validation distances. Published for exactly this use (spatial prediction into unsampled areas) and the natural upgrade of (c).
- *Geographic distance to the nearest station* (Phase M computes it anyway): a proxy for how far the transfer is, usable as a scale in the normalised methods above.

**8. Propagating the input uncertainty.** Perturb the inputs by their errors, run the model on each draw, and look at the spread of the outputs. The grid features carry `no2_sd` and `aai_sd`, but these are the spread of the source pixels inside the cell, not the retrieval error, so using them as an error would be a mislabelling. A proper version needs the error estimates that the data providers publish (for TROPOMI, per-pixel precision) and a model of the ERA5 error, neither of which has been gathered. Not used.

### G.4 Checking an uncertainty estimate

An uncertainty estimate that is not tested is a guess about a guess. The tests, all of which can use the existing out-of-fold predictions and none of which needs new data:

- **Coverage:** the share of held-out observations inside their band, reported overall and then separated by fold, month, prediction level and distance to the nearest training station. Calibration that holds only on average is not calibration. (Step 3 of notebook 11 prints the by-fold table.)
- **Width:** the average band width, to confirm it is not calibrated only by being huge.
- **Scores that combine the two:** the interval score, the pinball loss for quantiles, or the continuous ranked probability score (Gneiting and Raftery, 2007) reward calibrated and sharp bands together and allow comparing strategies on one number.
- **Do the warnings track the error?** Compute the disagreement and the range count at the *stations*, out-of-fold (the forest's out-of-fold predictions are already in `oof_predictions.parquet`; the range count needs the limits of each fold's training stations), and correlate them (Spearman) with the absolute out-of-fold error. If they do not rank the errors, they are not warnings. This is the most useful check of (b) and (c) and has not been done.
- **Separate calibration data from evaluation data.** If the band offsets and the coverage test use the same residuals (as now), the pooled coverage is 90% by construction. Calibrate on some folds and test on others to get a real figure.

### G.5 Uncertainty of aggregates (Phase N)

The equity analysis sums or averages predicted exposure over many cells. Treating each cell's band as independent would make the uncertainty of the total far too small, because the errors of neighbouring cells are correlated (reason 7). Practical options, in order of effort:

1. **Sensitivity analysis:** repeat the equity computation with every prediction set to its lower band bound, then to its upper bound, and with the predictions shifted and scaled within plausible ranges. Report whether the conclusion (for example, that desert cells are poorer) survives. It needs no new model and gives a conservative answer.
2. **Block bootstrap over stations:** refit the model on resampled spatial blocks of stations, predict the grid, recompute the equity statistic, and report the spread. It also propagates the model uncertainty into the aggregate.
3. **Simulated correlated error fields** from a spatial error model (G.3, item 6), if that model turns out to be justified.

Whichever is used, rank statements (which cells, or which income groups, are higher) are more reliable than level statements, because the compression of high values biases levels but largely preserves order (Phase J, item 5).

### G.6 Where the project stands and the upgrade path

| Strategy | Reasons covered (G.1) | Misses | Cost | Status |
|---|---|---|---|---|
| Analytic standard errors | 3 (partly) | 1, 2, 4–7, and invalid assumptions | trivial | not used |
| Tree spread | 3 | 1, 2, 4–7, no coverage | trivial (forest only) | not used (Ridge) |
| Residual band, global | 1, 2, part of 4 | local width, 5–7 | trivial | **used, indicator (a)** |
| Model disagreement | warning for 2 and 4 | size of error, shared mistakes | one more model | **used, indicator (b)** |
| Feature-range count | warning for 4 | joint novelty, importance weighting | trivial | **used, indicator (c)** |
| Split / cross-conformal | as the band, with a guarantee | local width | low | next |
| Normalised or weighted conformal | local width, 4 | needs a good scale, unstable weights | moderate | later |
| Quantile regression | local width | extrapolation | moderate | later |
| Bayesian ridge | 1, 3 under assumptions | non-Gaussian, non-constant noise | low | optional |
| Kriging / GP on residuals | 7, distance dependence | needs spatial structure | high | only if the semivariogram supports it |
| Area of applicability | warning for 4 | no error size | moderate | next |
| Input propagation | 5 | needs error data | moderate | not planned |

Suggested order, each step small and testable on the existing out-of-fold predictions:

1. Extend the coverage check to level, month and distance to the nearest training station (G.4); report where the global band fails.
2. Check whether disagreement and the range count rank the out-of-fold errors (G.4). Drop an indicator that does not.
3. If the band is too narrow in some region or level, make it local with scaled residuals, using the better of the indicators or the distance to the nearest station as the scale (G.3(3), G.3(4)).
4. Replace indicator (c) with the dissimilarity index of Meyer and Pebesma (G.3(7)) if joint novelty matters.
5. Run the sensitivity analysis of G.5 for the equity result before the write-up.

### G.7 How to word it in the report

Say what was done and no more: the band is "the 90% range of the spatially cross-validated residuals, applied as one width on the log scale", the two indicators are "warning signs, not error estimates", and none of them is a calibrated prediction interval. State that the cells farthest from any station, where the stations say least, are exactly where the band is least reliable, and that conclusions about high values are drawn from ranks and thresholds, not levels.

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. The foundational book on conformal prediction.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094–1111. Split conformal prediction for regression.
- Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511. A readable overview with worked examples.
- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive inference with the jackknife+. *Annals of Statistics*, 49(1), 486–507. Cross-validation-based conformal intervals.
- Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *Advances in Neural Information Processing Systems* 32. Re-weighting when the prediction points differ from the calibration points.
- Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33–50. The original quantile regression paper.
- Meinshausen, N. (2006). Quantile regression forests. *Journal of Machine Learning Research*, 7, 983–999.
- Wager, S., Hastie, T., & Efron, B. (2014). Confidence intervals for random forests: The jackknife and the infinitesimal jackknife. *Journal of Machine Learning Research*, 15, 1625–1651. What the spread of a forest's trees does and does not measure.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378. Interval score, CRPS and why they reward calibrated and sharp predictions.
- Diggle, P. J., & Ribeiro, P. J. (2007). *Model-based Geostatistics*. Springer. Spatially correlated errors, kriging variance.
- Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution*, 12(9), 1620–1633. The dissimilarity index and area of applicability: the closest published relative of the out-of-range indicator in Phase L.
