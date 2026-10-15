import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from monitoring_deserts.features import deserts


def test_cell_area_shrinks_with_latitude_and_the_sphere_adds_up():
    area = deserts.cell_area_km2([0, 60], 1.0)
    assert area[0] == pytest.approx(111.195**2, rel=1e-3)
    assert area[1] / area[0] == pytest.approx(0.5, abs=0.01)
    radius = 6371.0088
    total = deserts.cell_area_km2(np.arange(-89.5, 90), 1.0).sum() * 360
    assert total == pytest.approx(4 * np.pi * radius**2, rel=1e-9)
    with pytest.raises(ValueError):
        deserts.cell_area_km2([89.99], 0.05)


def test_probability_higher_matches_mann_whitney():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 300), rng.normal(0, 1, 500)
    assert deserts.probability_higher(a, b) == pytest.approx(mannwhitneyu(a, b).statistic / (300 * 500))
    assert deserts.probability_higher([1, 1], [1, 1]) == 0.5
    assert np.isnan(deserts.probability_higher([], [1.0]))


@pytest.fixture
def cells():
    return dict(
        distance_km=np.array([5, 20, 30, 60, 120, 300.0]),
        area_km2=np.full(6, 10.0),
        population=np.array([100, 50, np.nan, 10, 10, 30.0]),
        values=np.array([10, 20, 30, 40, 50, np.nan]),
    )


def test_desert_summary_counts_area_population_and_values(cells):
    table = deserts.desert_summary(cells["distance_km"], [10, 50, 100], cells["area_km2"], cells["population"], cells["values"])
    assert table["n_desert_cells"].tolist() == [5, 3, 2]
    assert table["share_area"].tolist() == pytest.approx([5 / 6, 0.5, 2 / 6])
    assert table["desert_population"].tolist() == [100.0, 50.0, 40.0]
    assert table["desert_cells_without_population"].tolist() == [1, 0, 0]
    assert table["value_mean_desert"][0] == pytest.approx(35)
    assert table["value_popweighted_desert"][1] == pytest.approx(45)
    assert table["prob_desert_higher"][0] == 1.0


def test_desert_summary_with_nobody_in_the_desert_gives_nan(cells):
    table = deserts.desert_summary(cells["distance_km"], [1000], cells["area_km2"], cells["population"], cells["values"])
    assert table["n_desert_cells"][0] == 0 and np.isnan(table["value_mean_desert"][0])


@pytest.mark.parametrize("thresholds", [[10, 10], [0, 5], []])
def test_desert_summary_rejects_bad_thresholds(cells, thresholds):
    with pytest.raises(ValueError):
        deserts.desert_summary(cells["distance_km"], thresholds, cells["area_km2"])


def test_desert_summary_rejects_bad_inputs(cells):
    with pytest.raises(ValueError):
        deserts.desert_summary(-cells["distance_km"], [1], cells["area_km2"])
    with pytest.raises(ValueError):
        deserts.desert_summary(cells["distance_km"], [1], cells["area_km2"], population=-np.abs(cells["population"]))
