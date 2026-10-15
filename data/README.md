# Data layout

Nothing in this folder is committed to git except this file (the `.gitignore` excludes `data/`): the raw
downloads are large and some of them have licences that do not allow redistribution (see "Data and licences" in the
top-level README). Every sub-folder is produced by a notebook in `noteboox/`; run them in order.

| Folder | Content | Produced by |
|---|---|---|
| `raw/station_census/` | OpenAQ station candidates, active sensors, coverage curves | notebook 01 |
| `raw/ground_truth/` | raw PM2.5 measurements from OpenAQ | notebook 02 |
| `raw/satellite/` | station-level TROPOMI extractions, and `grid/` monthly grid rasters | notebooks 03, 10 |
| `raw/meteorology/` | ERA5 monthly reanalysis (NetCDF) | notebook 05 |
| `raw/population/` | WorldPop rasters and the Relative Wealth Index files | notebook 06 |
| `raw/boundaries/` | Natural Earth country boundaries (used for maps) | `scripts/make_report_assets.py` |
| `clean/` | cleaned tables per topic (`satellite/`, `meteorology/`, `population/`, `equity/`) | notebooks 02-06, 13 |
| `processed/` | the analysis grid, master tables, grid predictions, monitoring status | notebooks 04, 07, 10-12 |
| `models/` | fitted models (`ridge_best.joblib`, ...) and out-of-fold predictions | notebook 09 |

The paths are set in `configs/project_config.yaml`, relative to the `noteboox/` folder.
