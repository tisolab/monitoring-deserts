# Spatial cross-validation: the full reasoning

This page explains the reasoning behind the spatial cross-validation of notebook 08 (`noteboox/08_cross_validation.ipynb`, code in `src/monitoring_deserts/models/spatial_cv.py`): why cross-validation exists at all, why the ordinary version fails for this kind of data, why the fix is built the way it is, and what limitations remain after fixing it. The phase names (Phase H and so on) are the pipeline stages listed in [../pipeline.md](../pipeline.md); section numbers (D.1, D.2, ...) are kept from the original write-up so cross-references still read.

### D.1 Why cross-validation exists at all

A model can always fit its own training data well — that alone tells you nothing about whether it learned something general or just memorized the rows it saw. Cross-validation estimates genuine generalization by holding out some data, training only on the rest, and scoring the model on the held-out part as a stand-in for data it has never seen. Everything on this page is really about how faithful a "stand-in" that held-out data actually is.

### D.2 The assumption ordinary random k-fold relies on

For a held-out fold to be a fair stand-in for new data, it needs to be statistically exchangeable with the training data — drawn independently from the same distribution, with no special relationship to any particular training point (the textbook i.i.d. assumption: independent and identically distributed). Random k-fold shuffles rows and splits them; if i.i.d. holds, a random fold is a fair simulation of unseen data and the resulting score is a fair estimate.

### D.3 Why that assumption breaks for station data: spatial autocorrelation

Air pollution measurements are not independent draws. Two stations 2 km apart share the same wind patterns, the same boundary-layer conditions, often the same nearby traffic corridor or industrial zone. This is a well-established property of geographic data with a name — Tobler's First Law of Geography: near things are more related than distant things. Two nearby stations' PM2.5 readings, TROPOMI values, and ERA5 values are all correlated with each other, not independent.

### D.4 The leakage mechanism, concretely

Suppose stations A and B are 2 km apart, and a random split puts A in training and B in test. The model doesn't need to learn the true physical relationship between satellite/weather features and PM2.5 to score well on B — it can partly get away with something closer to memorization: satellite/weather readings resembling A's tend to read about as much PM2.5 as A did, where "resembling" implicitly encodes local, place-specific information that leaked in through A's proximity rather than genuine physical signal. The model looks like it generalized. It didn't — it was scored on a test that was never really independent of its training data.

### D.5 Why this specific failure mode matters enormously for this project

The deployment task is exactly the scenario random CV never tests. The entire point of the project is predicting into grid cells with no nearby station at all — monitoring deserts, by definition, are far from any training point. A random-CV test point, by contrast, sits close to several training points with high probability, given realistic station density, precisely because random shuffling ignores geography entirely. Random CV quietly answers "how well can this model interpolate in places surrounded by good data," while the actual deliverable needs an answer to "how well can this model extrapolate into places with no data nearby." Reporting the first number as if it answers the second is a real, not merely academic, overconfidence risk for a project whose output could inform monitor-placement decisions.

### D.6 The fix: spatial blocking

Instead of scattering held-out points evenly through the training data, spatial block CV holds out an entire contiguous *region* at once — every station in that region moves to the test set together. This directly recreates the deployment scenario: the model is forced to predict for a chunk of geography where it has seen zero training examples, a genuine test of extrapolation rather than interpolation.

**Random k-fold vs. spatial block CV, illustrated:**

*(See the accompanying diagram: two panels of the same sixteen station locations. In random k-fold, the five held-out test stations are scattered among training stations, each sitting close to several training neighbors. In spatial block CV, the same five stations are grouped into one contiguous region, separated from every training station by a visible gap. The gap is the entire point — it's what forces genuine extrapolation rather than interpolation.)*

### D.7 Why KMeans on coordinates, specifically

Some method is needed to carve stations into contiguous regions before leaving them out block by block. KMeans clustering on raw latitude/longitude is the simplest option: it needs no external data source (no shapefiles, no administrative boundary lookups), is trivial to implement and inspect, and reliably produces geographically compact groups from coordinates alone. The real alternative — used in the larger version of this project for France and Spain — is holding out real administrative regions (NUTS2 units) one at a time, which has a genuine advantage: "we held out the whole of Île-de-France" is more interpretable to a policy reader than "we held out cluster 4." For a single-city or single-country pilot without a natural administrative hierarchy handy, KMeans is the pragmatic default — a design choice worth being explicit about in the write-up, not a hidden default.

### D.8 How many blocks, how many folds, and how big: matching the distances prediction actually requires

**Two dials, not one.** The number of *blocks* is how finely the stations are carved up. The number of *folds* is how many train/test rotations the model goes through. Earlier versions of this analysis tied them together (8 blocks, one per fold, "10-15 stations per block"). They are separate: `GroupKFold` can pack many small blocks into a few folds (D.9).

**What the earlier guidance said, and where it was only half right.** It was a bias-variance argument. Too few blocks make each one a huge, heterogeneous area, so the score averages over very few "measurements". Too many blocks (say 50, with 2-3 stations each) were said to resemble ordinary points, reintroducing leakage because a small block's neighbours sit in adjacent blocks, and to give test scores from too few points to be stable. The first worry is real for stations spread evenly across a region. It does not happen on this network (below). The second is about the number of *folds* and the rows per fold, not the number of blocks, and packing blocks into folds fixes it.

**The test that matters: distance to the nearest training station.** When the model predicts for an unmonitored grid cell, what it faces is the distance from that cell to the nearest station it was trained on. A cross-validation scheme is realistic when its held-out stations are about as far from the training stations as real prediction cells are from the stations. Random CV fails this badly, because a held-out station usually has a training station a few kilometres away. So the right way to choose block size is to measure that distance, not to follow a stations-per-block rule.

**What was measured** (2026-10-09; 343 stations, 35 blocks from `KMeans(n_init=15, random_state=42)` on the unique stations; "reality" is the 155,033 grid cells with population above zero in the five downloaded countries, each measured to the nearest of the 343 stations; all distances are great-circle):

| | folds | test rows per fold | held-out station to nearest training station: 10th / 25th / 50th / 75th / 90th percentile (km) |
|---|---|---|---|
| **Prediction reality** | - | - | 29 / 51 / **91** / 190 / 375 |
| Random `KFold(5)` over stations | 5 | 1,032-1,079 | 2 / 4 / **8** / 50 / 98 |
| Leave-one-block-out, 35 blocks | 35 | 36-611 | 90 / 105 / **134** / 175 / 227 |
| `GroupKFold(5)` over the 35 blocks | 5 | 1,025-1,059 | 91 / 107 / **144** / 218 / 258 |
| `GroupKFold(8)` over the 35 blocks | 8 | 638-671 | 91 / 106 / **139** / 184 / 231 |
| 8 KMeans regions, one per fold | 8 | 225-1,319 | 121 / 189 / **289** / 400 / 460 |

For the cells in "reality", 76% are farther than 50 km from the nearest station and 46% are farther than 100 km.

**How to read it:**
1. **Random CV is wrong by an order of magnitude** (median 8 km against 91 km). That gap is the inflation this project sets out to quantify.
2. **Packing small blocks into folds costs no spatial separation.** `GroupKFold` over the 35 blocks gives medians of 139-144 km, against 134 km for leaving out one block at a time, and the closest held-out-to-training pair across all folds is the same 46.5 km in both. This is because the stations form dense city clusters (the two largest blocks are Delhi, 40 stations, and Mumbai, 37) separated by large gaps, and KMeans follows those clusters. A block boundary does not run through the middle of a dense group of stations, so there is no seam to leak across.
3. **What changes is fold balance.** Leave-one-block-out has folds from 36 to 611 rows, so per-fold scores from the small ones are noisy, and it needs 35 model fits. `GroupKFold(5)` gives folds of about 1,045 rows each.
4. **The earlier default (8 regions, one per fold) is too harsh for typical cells:** its median of 289 km is about three times the real median, and its folds are uneven (225-1,319 rows; 15 to 85 stations per region).
5. **No scheme covers the whole range of reality.** The spatial schemes never test below about 90 km, while a quarter of real cells are within 51 km of a station; and they stop around 230-260 km at the 90th percentile, while reality reaches 375 km. The CV scores are therefore conservative for cells near a station and silent about the most remote ones. This goes into the limitations section (Phase O).

**Caveats.** One KMeans seed and one run, so the figures are indicative, and they should be recomputed if the station network or the blocks change. The check was run ad hoc and is not yet in the repository; it is worth adding to the cross-validation notebook as a diagnostic cell so the numbers can be reproduced. To use the same method on another station network: compute this table, pick the scheme whose median and 10th-90th range land near the prediction-cell distances, and among those prefer balanced folds.

### D.9 Why `GroupKFold`, specifically, as the mechanism

`GroupKFold` is the direct scikit-learn primitive for "given group labels, guarantee that no group ever appears in both train and test at once". It has two modes that matter here. When `n_splits` equals the number of groups, it is leave-one-block-out: each block is held out in turn. When `n_splits` is smaller (this project: 35 blocks, 8 folds), it packs whole blocks into folds so that the folds come out close to equal in size in rows (measured: 1,025-1,059 rows per fold for 5 folds, against 36-611 rows for leave-one-block-out). A block is never split between train and test in either mode. In the installed version (scikit-learn 1.9) it also accepts `shuffle=True` with a `random_state`, which changes which blocks share a fold; repeating the cross-validation with a few seeds shows how much the score depends on that arbitrary packing. Using `GroupKFold` directly, rather than hand-writing a loop that filters by cluster label, is less error-prone and immediately recognizable to anyone reviewing the code.

### D.10 Why a buffered split adds something plain blocking doesn't

Blocking alone still has a subtler leak at the seams: a station sitting right at the boundary between two KMeans clusters might be only a few hundred meters from a station just across that boundary in a different cluster — two clusters don't guarantee a real physical gap between them. A buffer (dead zone) explicitly drops training points within some distance of the test region, manufacturing a genuine spatial gap that plain blocking doesn't promise on its own. Running both the simple KMeans approach and the buffered approach isn't redundant — they test different failure modes, and agreement between the two is itself reassurance the result isn't an artifact of one specific method's blind spot.

**What the measurement showed for this project (2026-10-09).** The seam leak described above does not occur on the current blocks. Over every fold, the closest held-out station to any training station is 46.5 km away (D.8), so no training station lies within 25 km of a test station, and a 25 km buffer drops nothing; at 50 km it drops 2 stations. The buffered run is therefore only informative at radii of about 50 km or more, where it tests longer-range dependence rather than seam leakage, and at smaller radii it agrees with plain blocking by construction. A buffered split is not implemented in this repository (see [../known_issues.md](../known_issues.md)); it would only be informative at radii of about 50 km or more.

### D.11 Why random CV is still run at all, given it's the flawed one

Not to pick the better number — to **quantify the inflation**. "Random CV said R² of 0.75, spatial CV said R² of 0.45" is a concrete, defensible, quotable finding, and the single methodological line most likely to make a technical reader trust the rest of the write-up, because it demonstrates understanding the failure mode well enough to measure it rather than merely avoid it.

### D.12 The subtler trap: hyperparameter tuning, not just final evaluation

The general version of this concern is well-established, foundational machine learning methodology, not a niche point: Cawley & Talbot (2010) is the canonical paper showing that a biased or leaky model-selection criterion contaminates hyperparameter search itself, not only the final evaluation — the standard justification for nested cross-validation, cited directly by scikit-learn's own documentation. If hyperparameters are tuned with `RandomizedSearchCV` scored against ordinary random CV, and spatial CV is only used for the final reported number, the search process can systematically favor hyperparameters that best exploit local memorization, since that's what random-CV scoring rewards.

The spatial-specific version of this has been tested directly and empirically, and the result has a real nuance worth stating precisely rather than overselling. Schratz et al. (see the references at the end of this page) ran exactly this comparison — spatial vs. non-spatial hyperparameter tuning, across several models, on real spatial data — and found the effect on final performance varies by model: substantial for support vector machines, negligible for Random Forest and k-nearest-neighbors in their case study, while non-spatial *performance estimation* itself remained overoptimistic by 24–39% regardless of how tuning was done. In other words: tuning under spatial CV is well-founded methodological practice and worth doing as a matter of principle (Phase I passes the spatial `GroupKFold` splits as the `cv` argument to `RandomizedSearchCV` for exactly this reason), but its impact on this project's tree-based models (Random Forest, XGBoost) may turn out to be small rather than dramatic — that would be a legitimate, informative finding to report if it happens, not a sign anything went wrong.

### D.13 The residual limitation, even after all this

Spatial block CV is a large improvement over random CV, but not a perfect simulation of deployment. A held-out block's stations are, on average, some typical distance from the nearest training block — but the grid cells actually predicted into in Phase L are, almost by definition of being "monitoring deserts," farther from any training station than that. So even a well-designed spatial CV score is a mildly optimistic upper bound on performance in the very-farthest-from-any-station cells. This is the methodological reasoning underneath the existing limitation already flagged in Phase O — that predictions far from any station should be trusted less than the uncertainty band alone suggests — not a new caveat.

### D.14 Grounding in the literature, and one important nuance

Block cross-validation for spatially or temporally structured data is well established outside this project — especially in ecology, epidemiology, and remote sensing, anywhere geographically indexed data shows the same autocorrelation problem. The most-cited methodological reference is Roberts et al. (2017), which recommends block CV whenever dependence structure exists in the data, even if the fitted model already tries to account for that correlation directly. Full citations are in the references at the end of this page.

One nuance worth keeping, in the interest of not overclaiming: subsequent work has found that very strict spatial-independence methods (like buffered leave-one-out) can *overestimate* prediction error in some spatial-interpolation contexts — spatial CV isn't a universal "more correct" default for every geographic problem, it's the right choice specifically when the deployment task involves genuine extrapolation into unmonitored territory, which is precisely this project's situation. Worth being able to say plainly if a technically sophisticated reader questions the methodology.

## References

- Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881
- Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G. (2018). blockCV: An R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. *bioRxiv* 357798. The direct ecological-modeling ancestor of the `spacv` package, whose buffered-split idea Phase H implements in-project (Appendix D.15).
- Schratz, P., Muenchow, J., Iturritxa, E., Richter, J., & Brenning, A. Performance evaluation and hyperparameter tuning of statistical and machine-learning models using spatial data. arXiv:1803.11266. https://arxiv.org/abs/1803.11266 — the direct empirical test of spatial-vs-non-spatial hyperparameter tuning discussed in Appendix D.12.
- Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11(70), 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html — the canonical general (non-spatial) reference for why a biased model-selection criterion contaminates hyperparameter tuning itself, not just final evaluation.
