import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import box

from monitoring_deserts.viz import figures, webmap


def test_to_web_mercator_rows_is_identity_near_the_equator_and_shifts_rows_at_higher_latitude():
    raster = np.arange(100.0)[:, None] * np.ones((1, 3))
    near_equator = webmap.to_web_mercator_rows(raster, 1.0, -1.0)
    assert np.abs(near_equator - raster).max() <= 1
    shifted = webmap.to_web_mercator_rows(raster, 38.0, 5.0)
    assert shifted[0, 0] == 0 and shifted[-1, 0] == 99 and np.any(shifted != raster)
    # the middle output row (Mercator midpoint) samples the source near 22 degrees, not 21.5
    assert shifted[50, 0] < raster[50, 0] + 1
    with pytest.raises(ValueError):
        webmap.to_web_mercator_rows(raster, 5.0, 38.0)
    with pytest.raises(ValueError):
        webmap.to_web_mercator_rows(raster[0], 38.0, 5.0)


@pytest.fixture
def toy_inputs():
    rng = np.random.default_rng(0)
    shape = (40, 50)
    mask = np.ones(shape, dtype=bool)
    mask[:, :5] = False
    pm25 = rng.uniform(20, 60, shape)
    distance = rng.uniform(0, 200, shape)
    stations = pd.DataFrame({"station_id": ["a", "b"], "longitude": [72.0, 74.0], "latitude": [12.0, 13.0], "n_months": [16, 8], "mean_pm25": [55.0, 40.0]})
    zones = pd.DataFrame({"rank": [1, 2], "lon": [73.0, 75.0], "lat": [12.5, 13.5], "population": [3e6, 1e6], "mean_pm25": [50.0, 45.0], "mean_distance_km": [120.0, 150.0]})
    borders = gpd.GeoDataFrame(geometry=[box(70, 10, 76, 15)], crs="EPSG:4326")
    return pm25, distance, mask, stations, zones, borders, [70.0, 77.0, 10.0, 15.0]


def test_hero_map_and_the_other_figures_write_png_files(tmp_path, toy_inputs):
    pm25, distance, mask, stations, zones, borders, extent = toy_inputs
    figures.hero_map(pm25, distance, mask, stations, borders, extent, 100, tmp_path / "hero.png", "Title", "Subtitle")
    figures.cv_ladder({"constant": -0.09, "IDW": 0.12, "Ridge (same features)": 0.27, "Random Forest": 0.27}, 0.6, tmp_path / "ladder.png", "T", "S")
    shares = pd.DataFrame(np.linspace([0.9, 0.7, 0.4, 0.1], [0.4, 0.3, 0.2, 0.1], 5), columns=[25, 50, 100, 200])
    figures.equity_share_by_group(shares, tmp_path / "share.png", "T", "S")
    gaps = pd.DataFrame([{"proxy": p, "threshold_km": t, "estimate": 0.1, "lower": 0.0, "upper": 0.2} for p in ("RWI", "night light per person") for t in (25, 50, 100, 200)])
    correlations = pd.DataFrame({"label": ["a", "b", "c"], "estimate": [-0.2, -0.08, 0.01], "lower": [-0.3, -0.12, -0.03], "upper": [-0.18, -0.02, 0.05]})
    figures.equity_gap_and_correlations(gaps, correlations, tmp_path / "gap.png", "T", "S")
    figures.uncertainty_panels({i: 0.9 for i in range(8)}, 0.9, {"none": 0.2, "one or two": 0.3, "three or more": 0.5}, tmp_path / "unc.png", "T", "S")
    for name in ("hero", "ladder", "share", "gap", "unc"):
        assert (tmp_path / f"{name}.png").stat().st_size > 5000


def test_build_map_writes_a_self_contained_html_page(tmp_path, toy_inputs):
    pm25, distance, mask, stations, zones, _, extent = toy_inputs
    shown = np.where(mask, pm25, np.nan)
    path = tmp_path / "map.html"
    webmap.build_map(shown, distance > 100, extent, stations, zones, path, 100, 20, 60, "Title", "Subtitle", repo_url="https://example.org")
    html = path.read_text()
    assert "data:image/png;base64," in html and "Station a" in html and "Rank 1" in html and "https://example.org" in html
    with pytest.raises(ValueError):
        webmap.build_map(shown, (distance > 100)[:5], extent, stations, zones, path, 100, 20, 60, "T", "S")
    with pytest.raises(ValueError):
        webmap.build_map(shown, distance > 100, extent, stations, zones, path, 100, 60, 20, "T", "S")
