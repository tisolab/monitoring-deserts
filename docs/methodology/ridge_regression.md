# Ridge regression: the definitive model, explained

The phase names used below (Phase I, Phase L and so on) are the pipeline stages listed in [../pipeline.md](../pipeline.md); section numbers are kept from the original write-up so cross-references between the pages read correctly.


Phase I (model training, notebook 09) made a Ridge regression the definitive model, because it tied the Random Forest on the spatial folds and is simpler. This page explains what it is, why the penalty helps, and what it means for how this project's predictions should be read. It is written for someone comfortable with linear algebra and least squares, and no prior familiarity with machine learning.

### F.1 The problem it solves

Ordinary least squares (OLS) fits a linear model $y \approx Xw + b$ by choosing the coefficients that minimise the sum of squared errors on the training rows. With enough rows and features that are not strongly related to each other, that is a good fit. It goes wrong in two situations that both occur, to different degrees, in this project:

- **Correlated features.** Several of the thirteen features carry overlapping information: `relative_humidity` is computed from `temperature` and `dew_point`, `wind_magnitude` from `u_wind` and `v_wind`, `boundary_layer_height` and `temperature` move together seasonally. When features are nearly linear combinations of each other, OLS can only pin down the combination, not the individual coefficients: many very different coefficient vectors fit almost equally well, so the individual values swing wildly with small changes in the data (large positive on one feature, large negative on its twin).
- **Few rows relative to what the model can fit.** The more freedom a model has, the more it fits noise, and the worse it does on data it has not seen. Here the model is cheap (13 coefficients and an intercept for about 5,200 station-months), so this is the milder problem. But the station-months are not 5,200 independent observations, since a station's months are correlated, so the effective sample size is smaller than the row count.

Ridge keeps the same model and changes only how the coefficients are chosen, by adding a price for large coefficients. A worked example of each case, with the variance formulas, is in F.12.

### F.2 The model and the loss

The model is linear: $\hat y = b + w_1 x_1 + \dots + w_p x_p$. Ridge chooses $w$ (and the intercept $b$) to minimise

$$
\sum_{i=1}^{n} \left(y_i - b - x_i \cdot w\right)^2 \;+\; \alpha \left(w_1^2 + w_2^2 + \dots + w_p^2\right)
$$

The first term is the usual squared error. The second is the **penalty**: $\alpha$ (called `alpha` in scikit-learn, $\lambda$ in most textbooks) times the sum of the squared coefficients, which is the squared Euclidean (L2) length of $w$. Two details matter:

- **The intercept is not penalised.** Shifting the whole prediction up or down should cost nothing; only the coefficients, which say how strongly each feature moves the prediction, are penalised.
- **$\alpha$ is a dial between two extremes.** At $\alpha = 0$ this is exactly OLS. As $\alpha$ grows the coefficients shrink towards zero, and in the limit the model predicts only the training mean (the intercept). Everything in between trades a little extra training error for smaller, more stable coefficients.

Note on conventions: scikit-learn's `Ridge` uses exactly the objective above (no division by the number of rows), so its `alpha` is comparable between datasets of different sizes only if the rows are counted. Some textbooks divide the first term by `n` or by `2n`, which rescales $\alpha$; compare $\alpha$ values across sources with care.

### F.3 The closed-form solution

Ridge has an exact solution, with no iteration needed. For centred data (the intercept removed by subtracting the column means) it is

$$
w = \left(X^\top X + \alpha I\right)^{-1} X^\top y
$$

Compare OLS, $w = (X^\top X)^{-1} X^\top y$. The only difference is the $\alpha I$ added to $X^\top X$. When features are nearly collinear, $X^\top X$ is close to singular (it has an eigenvalue near zero) and its inverse blows up, which is the instability described in F.1. Adding $\alpha$ to the diagonal lifts every eigenvalue by at least $\alpha$, so the matrix is always invertible and the solution is stable. This is the original motivation of Hoerl and Kennard (1970): "ridge" refers to adding a ridge along the diagonal.

### F.4 What the penalty does: the singular-value view

Write the centred data matrix as $X = U D V^\top$ (singular value decomposition). The columns of $V$ are the directions in feature space along which the data vary independently ("principal components"), and the singular values $d_j$ say how much the data vary along each one. Then:

- OLS fits each direction fully.
- Ridge fits direction $j$ multiplied by the **shrinkage factor** $\dfrac{d_j^2}{d_j^2 + \alpha}$.

Directions with large $d_j$ (where the data vary a lot, so the coefficient is well determined) have a factor near 1 and are barely touched. Directions with small $d_j$ (where the data barely vary, which is exactly where collinearity makes OLS unstable) have a factor near 0 and are shrunk heavily. So ridge shrinks the poorly determined directions and leaves the well determined ones, which is why it helps when features are correlated and costs little when they are not.

A one-feature example makes the effect concrete. With one centred feature, $w = \dfrac{\sum_i x_i y_i}{\sum_i x_i^2 + \alpha}$: the OLS slope divided down by a factor that gets stronger as $\alpha$ grows relative to the spread of the feature. With many rows, $\sum_i x_i^2$ is large and a small $\alpha$ changes almost nothing.

### F.5 Bias, variance and why shrinking can lower the test error

Shrinking a coefficient away from its OLS value introduces **bias** (the model no longer fits the training data as well as it could) and reduces **variance** (the fitted coefficients vary less from one training sample to another). The error on new data is, roughly, bias squared plus variance plus irreducible noise. For small $\alpha$ the variance falls faster than the bias rises, so the prediction error on unseen data decreases; for large $\alpha$ the bias dominates and it rises again. The best $\alpha$ sits at the bottom of that U-shaped curve, and it is found by cross-validation (F.7). If the data are plentiful and the features not too collinear, the bottom of the curve is near $\alpha = 0$ and ridge is almost the same as OLS.

A Bayesian reading gives the same picture: ridge is the most probable coefficient vector (the posterior mode) if every coefficient is given an independent zero-mean Gaussian prior, with $\alpha$ encoding how narrow the prior is relative to the noise. A large $\alpha$ says "I expect every feature to have a small effect".

### F.6 Why the features must be standardised

The penalty adds up squared coefficients, so it treats all coefficients as comparable. They are not, if the features are on different scales. NO2 column density is of order $10^{-5}$ mol/m², surface pressure of order $10^{5}$ Pa, and a coefficient on NO2 has to be about ten orders of magnitude larger than the one on pressure to have the same effect on the prediction, so the penalty would crush the NO2 coefficient and barely notice the pressure one. Putting the features on a common scale (subtract the mean, divide by the standard deviation) makes the penalty even-handed: every coefficient is then "the change in the prediction per one standard deviation of that feature".

In this project the scaler is the first step of a scikit-learn `Pipeline`: `make_pipeline(StandardScaler(), Ridge(alpha=…))`. That is not only convenience. Inside a cross-validated search, the pipeline refits the scaler on each fold's training rows only, so the held-out region's mean and spread never leak into the scaling. A scaler fitted once on all rows before the cross-validation would be a small leak of exactly the kind the spatial folds exist to prevent ([spatial cross-validation](spatial_cv.md)).

### F.7 Choosing $\alpha$ in this project

`alpha` is a hyperparameter, not something the fit can choose, because the training error always prefers a smaller $\alpha$. It is chosen with `GridSearchCV` over a log-spaced grid, scored by negative root mean squared error on the **spatial** folds (the same convention as the two tree models, Phase I), so the choice is made on prediction in regions the training data did not contain.

Two things to check after the search, both recorded in `configs/model_training_metadata.yaml` under `definitive_model`:

- **Is the best $\alpha$ inside the grid or on its edge?** The best value found was the smallest in the grid (0.01), flagged as `alpha_at_grid_edge`. It means the search wanted less penalty than the grid offered, and the fit is essentially OLS: with this many rows and these features, shrinkage did not help on the spatial folds. This does not make Ridge the wrong choice (it is a tie with the forest and the simplest model), but a wider grid downwards would show whether the curve is flat or still falling, and it is listed in [../known_issues.md](../known_issues.md).
- **Is the score curve flat?** If many $\alpha$ values give nearly the same score, the choice is not important and the coefficients are stable across them.

Because the same spatial folds both choose $\alpha$ and report the score, the reported score is slightly optimistic, as for the other models ([spatial cross-validation](spatial_cv.md), D.12).

### F.8 Reading the coefficients

The standardised coefficients of the saved model (`standardised_coefficients_log1p` in the metadata) are the most readable output of this model, with these cautions:

- **Units.** Each coefficient is the change in the *log1p* prediction per one standard deviation of that feature, holding the others fixed. For PM2.5 well above a few µg/m³, a change of $c$ in $\log(1 + \mathrm{PM}_{2.5})$ corresponds to a multiplicative change of about $e^{c}$, a $100\,(e^{c} - 1)$ percent change, so coefficients read as roughly percentage effects (a coefficient of 0.1 is about +10%). It is not an additive µg/m³ effect.
- **Collinearity.** Where features overlap (temperature, dew point and relative humidity; u, v and wind magnitude; the seasonal cycle shared by many), a single coefficient is not reliable on its own, and the signs of correlated features can offset each other. Compare groups of related features, not single ones. Ridge makes this less severe than OLS but does not remove it.
- **Association, not cause.** A positive NO2 coefficient says that, across the station-months, higher column NO2 goes with higher PM2.5 once the other features are accounted for. It does not say NO2 produces PM2.5.
- **Cross-check.** The permutation importance of Phase J (what happens to the held-out error when a feature is scrambled) should agree with the coefficients on which features matter. Disagreement is worth reporting.

### F.9 How it behaves when predicting the grid

- **It extrapolates linearly.** A Random Forest can only return averages of training values, so its predictions stay within the training range and flatten at the extremes (the compression seen in the high-pollution fold, Phase J item 5). A linear model keeps going: a feature value beyond the training range moves the prediction proportionally further. That is an advantage when the true relationship really is close to linear beyond the data and a hazard when it is not. In the cleanest cells (low NO2, low aerosol index, below the stations' range) the prediction can go below zero on the original scale, which is why Phase L clips at zero and counts the rows. The `n_features_out_of_range` indicator marks the rows where this matters.
- **It has no interactions.** The effect of NO2 is the same whatever the boundary layer height, but physically a column of NO2 means more at the surface when the boundary layer is shallow. A linear model cannot represent that unless the interaction (or a transformed feature) is added by hand. A tree model finds such interactions on its own, which is one reason it could in principle beat Ridge; on the spatial folds it did not.
- **It has no memory of place.** Like every model here it uses no coordinates, so two cells with the same features get the same prediction wherever they are. Neighbouring cells are only similar through their features.
- **Its uncertainty is not part of the fit.** Ridge gives a point prediction. The band in Phase L is a global residual band, not a prediction interval, and the standard errors a statistics package would print for OLS do not apply to a penalised fit.

### F.10 When ridge is the right tool, and what to try next

Ridge is a good default when the relationship is roughly linear after a sensible transform, the features are correlated, the data are limited or spatially clustered, and a readable model matters. It is a poor choice when strong interactions or thresholds drive the outcome. Natural extensions, none done here:

- **Lasso or elastic net** (an L1 penalty, which sets some coefficients exactly to zero) to select features.
- **Interaction and transformed features** (for example NO2 × boundary layer height, or the log of NO2) while keeping the model linear and readable.
- **A rank metric and calibration.** Compare Spearman correlation per fold, and calibrate the predictions with a simple monotone map, since the monitoring-desert use needs the ranking more than the level.
- **Quantile regression**, or conformal prediction on the spatial folds, for a prediction interval with a stated coverage instead of a global residual band.

### F.11 Checklist before trusting a ridge fit

1. The scaler is inside the pipeline, so cross-validation refits it on each training fold.
2. $\alpha$ was chosen on spatial folds, and its position in the grid was checked (edge or interior).
3. The target transform is accounted for: coefficients and predictions are on the log1p scale until back-transformed.
4. Collinear features were looked at as groups, not read one by one.
5. Predictions outside the training feature range are flagged, and impossible values (negative concentrations) are clipped and counted.
6. Claims about the highest values are made about rank, not level.


### F.12 Worked example: how the penalty addresses the two pitfalls

F.1 named two situations where ordinary least squares misbehaves. This section shows, for each, what goes wrong and what the penalty changes, with a small example for the first and the variance formulas for the second.

**Pitfall 1: correlated features make individual coefficients unstable.** Suppose two features are nearly identical, so $x_2 \approx x_1$, and the true relation is $y = x_1 + x_2 + \text{noise}$. In the data only the sum of the two coefficients is visible: any pair with $w_1 + w_2 \approx 2$ fits almost equally well. OLS cannot choose among them and takes whichever the noise of this particular sample favours, for example $(3.2, -1.2)$ on one sample and $(-0.8, 2.8)$ on another (made-up numbers, to show the effect). The predictions are fine, since both pairs sum to 2, but the coefficients are meaningless and swing from sample to sample.

Ridge adds the cost $\alpha(w_1^2 + w_2^2)$. Among the pairs that fit equally well, which share the same sum $s$, the cheapest is the equal split $w_1 = w_2 = s/2$:

| pair | $w_1 + w_2$ | $w_1^2 + w_2^2$ |
|---|---|---|
| $(1,\ 1)$ | 2 | 2 |
| $(3.2,\ -1.2)$ | 2 | 11.7 |
| $(100,\ -98)$ | 2 | 19,604 |

So ridge returns about $(1, 1)$ whatever the noise: the data determine the sum, which they can, and the penalty decides how to split it, which the data cannot. In matrix terms, $X^\top X + \alpha I$ replaces $X^\top X$: the eigenvalue that was close to zero is lifted by $\alpha$, and its inverse no longer blows up (F.3).

**Pitfall 2: noise is fitted when the data are limited.** OLS finds the coefficients that best explain the training rows, including the part of $y$ that is noise. A different sample has different noise and so different coefficients; the spread of the estimates between samples is the variance of the fit. In the coordinates given by the singular value decomposition of F.4, with $\sigma^2$ the noise variance and $d_j^2$ measuring how much the data vary along direction $j$, the OLS estimate along that direction has

$$\operatorname{Var}\!\left(\hat w_j^{\text{OLS}}\right) = \frac{\sigma^2}{d_j^2}, \qquad \text{bias} = 0 .$$

The variance is large when there are few rows (all the $d_j^2$ are small) and in directions where the data hardly vary (a small $d_j^2$, which is the collinear case above). OLS is unbiased, but an unbiased estimate with a large variance is far from the truth in any one sample. Ridge shrinks each direction by the factor $d_j^2/(d_j^2+\alpha)$ (F.4), which gives

$$\operatorname{Var}\!\left(\hat w_j^{\text{ridge}}\right) = \frac{\sigma^2\, d_j^2}{\left(d_j^2 + \alpha\right)^2}, \qquad \text{bias} = -\frac{\alpha\, w_j}{d_j^2 + \alpha}.$$

The variance falls in every direction, most where $d_j^2$ is small, and a bias towards zero appears, most where $d_j^2$ is small too. For a suitable $\alpha$ the variance saved exceeds the squared bias added, so the squared error on new data is lower than OLS's. That is the U-shaped curve of F.5, and cross-validation finds its bottom (F.7).

**One mechanism.** The two pitfalls are the same thing seen from two sides: directions of the feature space about which the data say little, either because few rows were observed or because features are redundant. Ridge leaves the well-determined directions almost untouched and shrinks the poorly determined ones.

**How much this matters in this project.** Little for the second pitfall: 13 features and about 5,200 station-months leave all the $d_j^2$ large compared with the smallest value of $\alpha$ tried. The search chose $\alpha = 0.01$, the smallest in its grid, so the fitted model is close to OLS. Ridge was kept because it tied the forest and is the simplest model, and because it extrapolates, not because the penalty rescued a failing OLS fit. The first pitfall is real in this feature set (temperature, dew point and relative humidity; $u$, $v$ and wind magnitude), so individual coefficients are read in groups (F.8), and a small $\alpha$ only partly stabilises them. A direct check is to refit over a wider grid of $\alpha$ and watch the coefficients of temperature and dew point settle as $\alpha$ grows; the wider grid is listed in [../known_issues.md](../known_issues.md).


## References

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67. The original paper; introduces the penalty and the "ridge trace".
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), Springer, Section 3.4 (shrinkage methods), Section 7 (model assessment and selection). The standard textbook treatment, including the singular-value view used in F.4.
- scikit-learn documentation, "Linear Models" user guide, section "Ridge regression and classification", and the `sklearn.linear_model.Ridge` API page, for the exact objective and solvers used here (check the version installed in `mon-des`).
