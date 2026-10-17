#!/usr/bin/env python
"""Build the figures, the key numbers and the interactive map for the README and the write-up.

Run after the notebooks have produced their outputs (it reads `data/` and `configs/`):

    python scripts/make_report_assets.py

Writes:
    reports/figures/*.png     the figures used in README.md and reports/writeup.md
    reports/key_numbers.yaml  every number quoted in those documents, read from the saved metadata
    docs/index.html           the summary page served by GitHub Pages
    docs/map.html             the interactive map (also previewed on the summary page)
    docs/figures/*.png        copies of the figures, so the summary page can serve them

The maps show the five countries with population data (India, Pakistan, Bangladesh, Nepal and
Sri Lanka), masked with Natural Earth boundaries; the analysis grid itself covers the whole
bounding box of the study region.
"""

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from monitoring_deserts.data import fetch_boundaries
from monitoring_deserts.features import deserts
from monitoring_deserts.viz import figures, landing, webmap

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "noteboox"  # the paths of project_config.yaml are relative to this folder
DESERT_KM = 100
THRESHOLDS_KM = [25, 50, 100, 200]
PAGE_TITLE = "Monitoring deserts in South Asia"


def load_yaml(name):
    with open(ROOT / "configs" / name) as f:
        return yaml.safe_load(f)


def resolve(path):
    return (NOTEBOOK_DIR / path).resolve()


def main(repo_url):
    config = load_yaml("project_config.yaml")
    processed, clean, raw = (resolve(config["data_dir"][k]) for k in ("processed", "clean", "raw"))
    cell_size = config["grid_cell_size_deg"]
    figure_dir, docs_dir = ROOT / "reports" / "figures", ROOT / "docs"
    figure_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(exist_ok=True)

    model = load_yaml("model_training_metadata.yaml")
    equity = load_yaml("equity_metadata.yaml")
    desert_meta = load_yaml("monitoring_deserts_metadata.yaml")
    predictions_meta = load_yaml("grid_predictions_metadata.yaml")
    validation = load_yaml("validation_metadata.yaml")

    # --- tables on the grid
    grid = gpd.read_parquet(processed / "grid.parquet")[["cell_id", "centroid_lon", "centroid_lat"]]
    population = pd.read_parquet(clean / "population" / "grid_features.parquet", columns=["cell_id", "population"])
    status = pd.read_parquet(processed / "cell_monitoring_status.parquet", columns=["cell_id", "distance_to_station_km"])
    monthly = pd.read_parquet(processed / "grid_predictions.parquet", columns=["cell_id", "month", "predicted_pm25"])
    monthly["month_of_year"] = monthly["month"].dt.month
    # The period the predictions describe, read from the data (not typed in)
    period_start, period_end = monthly["month"].min(), monthly["month"].max()
    period_label = f"{period_start.strftime('%B %Y')} to {period_end.strftime('%B %Y')}"
    n_months = int(monthly["month"].nunique())
    balanced = monthly.groupby(["cell_id", "month_of_year"])["predicted_pm25"].mean().groupby("cell_id").mean().rename("pm25")
    monthly_pivot = monthly.pivot(index="cell_id", columns="month", values="predicted_pm25")  # one column per month, for the month selector
    del monthly  # each calendar month averaged over the years, then the 12 months averaged
    cells = grid.merge(population, on="cell_id").merge(status, on="cell_id").merge(balanced, on="cell_id", how="left")

    grid_lons = np.sort(cells["centroid_lon"].round(9).unique())
    grid_lats = np.sort(cells["centroid_lat"].round(9).unique())
    col = np.searchsorted(grid_lons, cells["centroid_lon"].round(9).values)
    lat_idx = np.searchsorted(grid_lats, cells["centroid_lat"].round(9).values)

    def raster(values, fill=np.nan):
        out = np.full((len(grid_lats), len(grid_lons)), fill, dtype=float)
        out[lat_idx, col] = values
        return out

    # --- study area: the five countries, plus any populated cell on the coast the 50 m boundaries miss
    boundaries = fetch_boundaries.download_country_boundaries(raw / "boundaries")
    countries = fetch_boundaries.load_countries(boundaries)
    mask = fetch_boundaries.country_mask(grid_lons, grid_lats, countries) | (raster(cells["population"].to_numpy()) > 0)
    rows, cols = np.where(mask)
    margin = 5
    r0, r1 = max(rows.min() - margin, 0), min(rows.max() + margin, len(grid_lats) - 1)
    c0, c1 = max(cols.min() - margin, 0), min(cols.max() + margin, len(grid_lons) - 1)
    extent = [grid_lons[c0] - cell_size / 2, grid_lons[c1] + cell_size / 2, grid_lats[r0] - cell_size / 2, grid_lats[r1] + cell_size / 2]

    def northup(array):  # crop to the study area and flip to a north-up raster (row 0 northernmost)
        return array[r0 : r1 + 1, c0 : c1 + 1][::-1]

    pm25 = northup(raster(cells["pm25"].to_numpy()))
    distance = northup(raster(cells["distance_to_station_km"].to_numpy()))
    study = northup(mask)
    month_rasters = {  # predicted PM2.5 of each month, labelled for the selector on the map
        month.strftime("%B %Y"): np.where(study, northup(raster(monthly_pivot[month].reindex(cells["cell_id"]).to_numpy())), np.nan)
        for month in monthly_pivot.columns
    }

    station_months = pd.read_parquet(processed / "master_station_df.parquet", columns=["station_id", "longitude", "latitude", "pm25"])
    stations = (
        station_months.groupby("station_id")
        .agg(longitude=("longitude", "first"), latitude=("latitude", "first"), n_months=("pm25", "count"), mean_pm25=("pm25", "mean"))
        .reset_index()
    )
    stations = stations[stations["n_months"] > 0]
    in_view = stations["longitude"].between(extent[0], extent[1]) & stations["latitude"].between(extent[2], extent[3])
    stations = stations[in_view]

    # --- numbers over the study mask (notebook 12 counted sea and other countries in its area shares)
    in_mask = mask[lat_idx, col]
    area = deserts.cell_area_km2(cells["centroid_lat"], cell_size)
    mask_summary = deserts.desert_summary(
        cells.loc[in_mask, "distance_to_station_km"].to_numpy(), THRESHOLDS_KM, area[in_mask],
        population=cells.loc[in_mask, "population"].to_numpy(),
    )

    # --- figures
    figures.hero_map(
        pm25, distance, study, stations, countries, extent, DESERT_KM, figure_dir / "hero_map.png",
        title="Where the monitors are, and where they are not",
        subtitle=f"Mean predicted PM2.5, {period_label} (a historical average, not a live reading). A model estimate: uncertain at single cells, and best read as a pattern.",
    )

    ladder = model["baseline_ladder"]["metrics_ugm3"]
    metrics = {name: values["pooled r2"] for name, values in ladder.items()}
    random_r2 = model["headline_metrics_ugm3"]["random_folds"]["r2"]
    spatial_r2 = model["headline_metrics_ugm3"]["spatial_folds"]["r2"]
    figures.cv_ladder(
        metrics, random_r2, figure_dir / "cv_ladder.png",
        title="Predicting a region the model never saw is much harder than random folds suggest",
        subtitle=f"The same forest scores R² {random_r2:.2f} on random folds and {spatial_r2:.2f} on held-out regions.",
    )

    shares = pd.DataFrame(equity["share_in_desert_by_group_and_threshold"]).set_index("index")
    shares.columns = [int(c.replace("> ", "").replace(" km", "")) for c in shares.columns]
    poorest, richest = shares.iloc[0], shares.iloc[-1]
    figures.equity_share_by_group(
        shares, figure_dir / "equity_desert_share.png",
        title="Share of each income group's people living far from a monitor",
        subtitle=f"Beyond 25 km: {poorest[25]:.0%} of the poorest fifth against {richest[25]:.0%} of the richest. At 200 km the groups are alike.",
    )

    gaps = pd.DataFrame(equity["poorest_minus_richest_desert_share"])
    corr = pd.DataFrame(equity["correlations"]).rename(columns={"index": "label"})
    corr["label"] = corr["label"].map({
        "Spearman, distance vs RWI": "no control",
        "partial, controlling density": "controlling population density",
        "partial, controlling density and night light": "controlling density and night light",
    })
    figures.equity_gap_and_correlations(
        gaps, corr, figure_dir / "equity_gap.png",
        title="Much of the income gap in distance to a monitor is an urban-versus-rural gap",
        subtitle="Left: the gap by threshold, with two income proxies. Right: the correlation shrinks towards zero once density and night light are controlled.",
    )

    coverage = predictions_meta["residual_band"]["share_inside_band_by_fold"]
    out_of_range = validation["out_of_range_share"]
    worst_fold = min(coverage, key=coverage.get)
    figures.uncertainty_panels(
        {int(k): v for k, v in coverage.items()}, predictions_meta["residual_band"]["level"],
        {"no feature outside": out_of_range["no feature outside (rows)"], "1 or 2 features outside": out_of_range["1 or 2 features outside"],
         "3 or more features outside": out_of_range["3 or more features outside"]},
        figure_dir / "uncertainty.png",
        title="One uncertainty band does not fit every region, and half the grid is unlike the stations",
        subtitle=f"The band holds {coverage[worst_fold]:.0%} of observations in fold {int(worst_fold)}; "
                 f"{out_of_range['3 or more features outside']:.0%} of predicted cell-months have 3 or more features outside the stations' range.",
    )

    # --- interactive map
    zones = pd.read_parquet(clean / "equity" / "desert_zones_ranked.parquet").head(25)
    low, high = np.nanquantile(np.stack(list(month_rasters.values())), [0.02, 0.98])  # one colour scale for every month
    pages_url = "https://{}.github.io/{}/".format(*repo_url.rstrip("/").split("/")[3:5])
    webmap.build_map(
        np.where(study, pm25, np.nan), study & (distance > DESERT_KM), extent, stations, zones, docs_dir / "map.html", DESERT_KM, low, high,
        title=PAGE_TITLE, subtitle="Predicted PM2.5, the 343 monitoring stations, and the zones where a new monitor would cover the most exposure.",
        repo_url=repo_url, home_url=pages_url, period_label=period_label, monthly=month_rasters,
    )

    # --- key numbers, each read from a saved metadata file
    definitive = model["definitive_model"]
    key = {
        "sources": "configs/*.yaml written by notebooks 09-14; area shares recomputed over the study-country mask by this script",
        "data": {
            "n_stations": model["n_stations"], "n_station_months": model["n_data_points"], "n_months": n_months, "period": period_label,
            "n_cells": desert_meta["n_cells"], "n_cells_study_area": int(in_mask.sum()), "grid_cell_size_deg": cell_size,
        },
        "cross_validation": {"spatial": model["headline_metrics_ugm3"]["spatial_folds"], "random": model["headline_metrics_ugm3"]["random_folds"]},
        "baseline_ladder_pooled_r2": metrics,
        "paired_forest_vs_baselines": model["baseline_ladder"]["paired_rmse_forest_minus_baseline"],
        "ridge": {"best_alpha": definitive["best_alpha"], "alpha_at_grid_edge": definitive["alpha_at_grid_edge"], "paired_rmse_ridge_minus_forest": definitive["paired_rmse_ridge_minus_forest"]},
        "coefficients_log1p": definitive["standardised_coefficients_log1p"],
        "desert_study_area": mask_summary.round(4).to_dict(orient="records"),
        "desert_notebook12_population_shares": {int(r["threshold_km"]): r["share_population"] for r in desert_meta["summary"]},
        "equity": {
            "share_in_desert_by_group_and_threshold": equity["share_in_desert_by_group_and_threshold"],
            "poorest_minus_richest": equity["poorest_minus_richest_desert_share"], "correlations": equity["correlations"],
            "sample": equity["sample"],
        },
        "uncertainty": {"band": predictions_meta["residual_band"], "out_of_range_share": out_of_range},
        "validation": {"seasonal_cycle": validation["seasonal_cycle_agreement"], "spot_checks": validation["spot_checks"], "validated_distance_km": validation["validated_distance_km"]},
    }
    with open(ROOT / "reports" / "key_numbers.yaml", "w") as f:
        yaml.safe_dump(key, f, sort_keys=False)

    # --- the summary page, with copies of the figures it shows
    (docs_dir / "figures").mkdir(exist_ok=True)
    for name in landing.FIGURES:
        shutil.copy(figure_dir / name, docs_dir / "figures" / name)
    landing.build_landing_page(key, docs_dir / "index.html", repo_url)
    print("Wrote", *sorted(p.name for p in figure_dir.glob("*.png")), "reports/key_numbers.yaml docs/index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo-url", default="https://github.com/tisolab/monitoring-deserts", help="link shown in the map's title box")
    main(parser.parse_args().repo_url)
