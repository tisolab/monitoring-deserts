# Known issues and open items

Real but non-blocking points, recorded rather than hidden. Items that affect how a result should be read are repeated in the README's limitations.

## Data

- **One spurious month in the ground truth.** The month grid of notebook 02 includes the end date's own month, so the ground-truth table has 17 months where the satellite windows cover 16. The trailing month (July 2026) is dropped in notebook 03. The cleaner fix is to derive both month ranges from one function (`fetch_earthengine.build_month_windows`).
- **July 2026 coverage.** That dropped month was unusual: 318 of its 344 stations were flagged as having under 60% coverage, against 160 such flags across all 16 kept months together. It looks like a reporting gap in the source data rather than ordinary missingness.
- **Population has no zero.** The population grid has no cell with a value of exactly zero: a cell is either populated or empty (NaN), and the empty cells cannot be told apart into sea, uninhabited land and land outside the five countries for which population was downloaded (India, Pakistan, Bangladesh, Nepal, Sri Lanka). About 64% of the grid is empty. Population-based statistics cover only the cells with data.
- **Station values versus cell values.** A station value is the mean over a 2 km buffer, a grid value the mean over a 0.05 degree cell (about 5 km). The two have different support; notebook 10 compares each station with the cell that contains it.
- **Relative Wealth Index.** It is distributed under CC BY-NC 4.0 (non-commercial) and is not redistributed here. It is a modelled product with its own error.

## Model

- **Ridge `alpha` at the edge of its grid.** The best value found was the smallest of the grid (0.01), so the fitted model is close to ordinary least squares. A wider grid downwards would show whether the score curve is flat; see the Ridge page in `docs/methodology/`.
- **No elevation feature.** The high-altitude spot check (Leh) is predicted in the middle of the distribution; weather features partly, but not fully, encode altitude.
- **The polluted fold.** Spatial fold 0 (including the Delhi region) has a per-fold R² below zero for every model; predictions there are compressed towards the middle although they rank the station-months well (correlation about 0.83 in a diagnostic refit). It is reported as a finding, not fixed.
- **Residual band.** The uncertainty band is one width for the whole grid; its coverage by fold ranges from 78% to 95% against an intended 90%. It is not a calibrated prediction interval. Upgrades (local or conformal bands, an applicability index) are described in `docs/methodology/uncertainty.md`.
- **Not built.** A buffered spatial cross-check (it would only be informative at radii of about 50 km or more) and a semivariogram of the model residuals.

## Maps and area figures

- **No land mask in the notebooks.** The analysis grid covers the whole bounding box of the study region, so it includes sea and land outside the five countries. Notebooks 11 to 14 predict for all of it, and the area-based desert shares of notebook 12 count all of it. The report figures and `reports/key_numbers.yaml` are restricted to the five countries with Natural Earth boundaries; population-based shares are unaffected.
- **Eastern edge.** The grid stops at 93.4 degrees east, so the maps cut off the far north-east of India.
