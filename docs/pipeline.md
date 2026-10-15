# The pipeline: phases, notebooks, code and outputs

The project is a sequence of fourteen notebooks in `noteboox/`, run in order. Reusable logic lives in `src/monitoring_deserts/` (imported by the notebooks, tested in `tests/`). Each notebook writes its outputs under `data/` (not in git) and a small metadata file under `configs/` (in git) recording its settings and headline numbers; `scripts/make_report_assets.py` reads those files to build the figures and the numbers quoted in the README and the write-up.

The "phase" letters are the stages of the original build plan; the methodology pages refer to them.

| Phase | Stage | Notebook | Code in `src/monitoring_deserts/` | Main outputs |
|---|---|---|---|---|
| A | Station census: which OpenAQ stations are active and long-lived enough | `01_station_census` | `data/fetch_openaq.py` | `data/raw/station_census/`, `configs/census_scope.yaml` |
| B | Ground-truth extraction: monthly PM2.5 per station | `02_ground_truth_extraction` | `data/fetch_openaq.py` | `data/raw/ground_truth/`, `data/clean/satellite/ground_truth_mod.parquet` |
| C | Satellite features at the stations (TROPOMI NO2 and aerosol index) | `03_satellite_features_extraction` | `data/fetch_earthengine.py` | `data/clean/satellite/analysis_df.parquet`, `configs/satellite_metadata.yaml` |
| D | The analysis grid (0.05 degree cells) | `04_analysis_grid` | | `data/processed/grid.parquet` |
| E | Meteorology (ERA5) at the stations and on the grid | `05_meteorology_features` | `data/fetch_era5.py` | `data/clean/meteorology/`, `configs/meteorology_metadata.yaml` |
| F | Population, wealth index and night-time light | `06_population_income_features` | `data/fetch_population.py` | `data/clean/population/`, `configs/population_income_metadata.yaml` |
| G | Assemble and explore | `07_exploration` | | `data/processed/master_station_df.parquet`, `configs/eda_metadata.yaml` |
| H | Spatial cross-validation set-up | `08_cross_validation` | `models/spatial_cv.py` | `data/clean/cross_validation/station_blocks_folds.parquet`, `configs/cross_validation_metadata.yaml` |
| I, J | Model training, baselines, evaluation and interpretation | `09_model_training` | `models/spatial_cv.py` | `data/models/*.joblib`, `configs/model_training_metadata.yaml` |
| K | Satellite features on the whole grid | `10_grid_satellite_features` | `data/fetch_earthengine.py` | `data/processed/master_grid_df.parquet`, `configs/grid_satellite_metadata.yaml` |
| L | Predict into every cell, with uncertainty indicators | `11_grid_predictions` | `models/predict.py` | `data/processed/grid_predictions.parquet`, `configs/grid_predictions_metadata.yaml` |
| M | Define and map the monitoring deserts | `12_monitoring_deserts` | `features/deserts.py` | `data/processed/cell_monitoring_status.parquet`, `configs/monitoring_deserts_metadata.yaml` |
| N | Equity analysis | `13_equity_analysis` | `features/equity.py` | `data/clean/equity/`, `configs/equity_metadata.yaml` |
| O | Validation and limitations | `14_validation_and_limitations` | | `configs/validation_metadata.yaml` |
| P | Packaging | `scripts/make_report_assets.py` | `data/fetch_boundaries.py`, `viz/` | `reports/`, `docs/index.html` |

## Conventions

- **Paths** are set once in `configs/project_config.yaml`, relative to the `noteboox/` folder, where the notebooks run.
- **Grid rasters** use one convention everywhere: the unique, rounded cell-centre longitudes and latitudes give the axes; a north-up raster has row 0 at the northernmost row, and a cell is read back by `np.searchsorted` on those axes, so tables are keyed on `cell_id` and never depend on row order.
- **The target** is `log1p(PM2.5)`; predictions are back-transformed with `np.expm1` before any number in micrograms per cubic metre is quoted.
- **Spatial folds** are built once (notebook 08) and reused everywhere; hyperparameters are tuned on them, never on random folds.
