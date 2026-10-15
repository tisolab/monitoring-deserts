import types

import numpy as np
import pandas as pd
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from monitoring_deserts.data import fetch_earthengine as fe

pytestmark = pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")


def test_build_month_windows_steps_one_calendar_month_from_the_start_day():
    windows = fe.build_month_windows("2025-03-15", "2025-06-15")
    assert [(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")) for s, e in windows] == [
        ("2025-03-15", "2025-04-15"), ("2025-04-15", "2025-05-15"), ("2025-05-15", "2025-06-15"),
    ]


def test_build_month_windows_validates_the_period():
    with pytest.raises(ValueError):
        fe.build_month_windows("2025-06-01", "2025-03-01")
    with pytest.raises(ValueError):
        fe.build_month_windows("not a date", "2025-03-01")


class FakeImage:
    """Stands in for an ee.Image: serves tiles of a known array and records its calls."""

    def __init__(self, data):
        self.data = data
        self.reducer = self.bands = None

    def reduceResolution(self, reducer, maxPixels):
        self.reducer = reducer
        return self

    def reproject(self, crs, crsTransform):
        self.transform = crsTransform
        return self

    def select(self, bands):
        self.bands = bands
        return self

    def getDownloadURL(self, params):
        left, top, size = self.transform[2], self.transform[5], self.transform[0]
        x_min, y_min, x_max, y_max = params["region"]
        col0, col1 = int(np.floor((x_min - left) / size)), int(np.ceil((x_max - left) / size))
        row0, row1 = int(np.floor((top - y_max) / size)), int(np.ceil((top - y_min) / size))
        return self.data[:, row0:row1, col0:col1]


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(fe.ee, "Geometry", types.SimpleNamespace(Rectangle=lambda bounds, **kw: tuple(bounds)))
    monkeypatch.setattr(fe.ee, "Reducer", types.SimpleNamespace(mean=lambda: "mean"))
    state = {"fail": 0, "bands": None}

    def get(url, timeout):
        if state["fail"]:
            state["fail"] -= 1
            raise fe.requests.ConnectionError("temporary")
        array = url if state["bands"] is None else url[: state["bands"]]
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", height=array.shape[1], width=array.shape[2], count=array.shape[0],
                             dtype="float32", nodata=-9999.0, transform=from_origin(0, 0, 1, 1)) as dst:
                dst.write(np.where(np.isnan(array), -9999, array).astype("float32"))
            content = memory.read()
        return types.SimpleNamespace(content=content, raise_for_status=lambda: None)

    monkeypatch.setattr(fe.requests, "get", get)
    monkeypatch.setattr(fe.time, "sleep", lambda seconds: None)
    return state


AXES = (np.round(70 + 0.05 * np.arange(23), 9), np.round(10 + 0.05 * np.arange(17), 9), 0.05)


def test_download_image_on_grid_single_band_keeps_the_north_up_layout(stubbed):
    full = np.random.default_rng(0).normal(size=(1, 17, 23))
    full[0, 3, 4] = np.nan
    out = fe.download_image_on_grid(FakeImage(full), *AXES, tile_cells=5, progress=False)
    assert out.shape == (17, 23)
    assert np.allclose(np.nan_to_num(out), np.nan_to_num(full[0]), atol=1e-6) and np.isnan(out[3, 4])


def test_download_image_on_grid_selects_bands_in_the_requested_order_and_in_parallel(stubbed):
    full = np.random.default_rng(1).normal(size=(2, 17, 23))
    image = FakeImage(full)
    out = fe.download_image_on_grid(image, *AXES, tile_cells=5, progress=False, reducer="R", bands=["a", "b"], max_workers=4)
    assert out.shape == (2, 17, 23) and np.allclose(out, full, atol=1e-6)
    assert image.bands == ["a", "b"] and image.reducer == "R"


def test_download_image_on_grid_retries_then_gives_up(stubbed):
    full = np.random.default_rng(2).normal(size=(2, 17, 23))
    stubbed["fail"] = 2
    out = fe.download_image_on_grid(FakeImage(full), *AXES, tile_cells=5, progress=False, bands=["a", "b"], max_workers=3, retries=2)
    assert np.allclose(out, full, atol=1e-6)
    stubbed["fail"] = 1
    with pytest.raises(RuntimeError, match="Failed downloading grid tile"):
        fe.download_image_on_grid(FakeImage(full), *AXES, tile_cells=5, progress=False, bands=["a", "b"], retries=0)


def test_download_image_on_grid_checks_band_count_and_arguments(stubbed):
    full = np.random.default_rng(3).normal(size=(2, 17, 23))
    stubbed["bands"] = 1
    with pytest.raises(RuntimeError, match="expected 2 band"):
        fe.download_image_on_grid(FakeImage(full), *AXES, progress=False, bands=["a", "b"])
    stubbed["bands"] = None
    for kwargs in ({"bands": []}, {"bands": ["a", "a"]}, {"max_workers": 0}, {"retries": -1}):
        with pytest.raises(ValueError):
            fe.download_image_on_grid(FakeImage(full), *AXES, progress=False, **kwargs)
    with pytest.raises(ValueError):
        fe.download_image_on_grid(FakeImage(full), AXES[0][::-1], AXES[1], AXES[2], progress=False)
