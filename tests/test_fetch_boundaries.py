import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import box

from monitoring_deserts.data import fetch_boundaries as fb


def test_country_mask_keeps_cells_whose_centre_is_inside():
    lons, lats = np.array([0.5, 1.5, 2.5]), np.array([0.5, 1.5])
    mask = fb.country_mask(lons, lats, box(0, 0, 2, 1))  # covers the two western cells of the southern row
    assert mask.tolist() == [[True, True, False], [False, False, False]]  # row 0 is the southernmost


def test_country_mask_accepts_a_geodataframe_and_validates_axes():
    gdf = gpd.GeoDataFrame({"ADM0_A3": ["AAA"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    assert fb.country_mask([0.5], [0.5], gdf).tolist() == [[True]]
    with pytest.raises(ValueError):
        fb.country_mask([1.0, 0.0], [0.5], gdf)


def test_load_countries_filters_by_code_and_rejects_unknown(tmp_path):
    gdf = gpd.GeoDataFrame({"ADM0_A3": ["AAA", "BBB"]}, geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)], crs="EPSG:4326")
    path = tmp_path / "countries.gpkg"
    gdf.to_file(path, driver="GPKG")
    assert fb.load_countries(path, ["aaa"])["ADM0_A3"].tolist() == ["AAA"]
    with pytest.raises(ValueError):
        fb.load_countries(path, ["ZZZ"])
    with pytest.raises(ValueError):
        fb.load_countries(path, [])


def test_download_is_skipped_when_the_file_exists(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return type("Response", (), {"content": b"zip", "raise_for_status": lambda self: None})()

    monkeypatch.setattr(fb.requests, "get", fake_get)
    path = fb.download_country_boundaries(tmp_path / "boundaries")
    assert path.read_bytes() == b"zip" and len(calls) == 1
    fb.download_country_boundaries(tmp_path / "boundaries")
    assert len(calls) == 1
    fb.download_country_boundaries(tmp_path / "boundaries", overwrite=True)
    assert len(calls) == 2
