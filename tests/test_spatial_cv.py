import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from monitoring_deserts.models import spatial_cv as cv


@pytest.fixture
def blocked(station_months):
    return cv.cluster_stations(station_months, stations_per_block=6)


def test_cluster_stations_gives_every_month_of_a_station_the_same_block(blocked):
    assert blocked.groupby("station_id")["spatial_block_id"].nunique().eq(1).all()
    assert blocked["spatial_block_id"].nunique() == 10  # ceil(60 / 6)


def test_cluster_stations_rejects_nulls_and_bad_arguments(station_months):
    with_null = station_months.copy()
    with_null.loc[0, "longitude"] = np.nan
    with pytest.raises(ValueError):
        cv.cluster_stations(with_null)
    with pytest.raises(ValueError):
        cv.cluster_stations(station_months, n_init=0)


def test_group_k_fold_never_splits_a_block_and_partitions_the_rows(blocked):
    splits = cv.group_k_fold(blocked, n_folds=5)
    assert len(splits) == 5
    all_test = np.concatenate([test for _, test in splits])
    assert sorted(all_test) == list(range(len(blocked)))
    for train, test in splits:
        train_blocks = set(blocked.iloc[train]["spatial_block_id"])
        test_blocks = set(blocked.iloc[test]["spatial_block_id"])
        assert not train_blocks & test_blocks


def test_group_k_fold_validates_input(blocked, station_months):
    with pytest.raises(ValueError):
        cv.group_k_fold(station_months)  # no block column
    with pytest.raises(ValueError):
        cv.group_k_fold(blocked, n_folds=11)  # more folds than blocks
    with pytest.raises(ValueError):
        cv.group_k_fold(blocked, n_folds=1)


def test_run_cross_validation_scores_a_linear_signal(blocked):
    X, y = blocked[["x1", "x2"]], blocked["y"]
    metrics, oof = cv.run_cross_validation(LinearRegression(), X, y, cv.group_k_fold(blocked, 5), inverse_transform=None)
    assert list(metrics.columns) == ["fold", "n_test", "r2", "rmse", "mae"]
    assert len(metrics) == 5 and not np.isnan(oof).any()
    assert metrics["r2"].min() > 0.9


def test_run_cross_validation_leaves_rows_outside_every_test_fold_as_nan(blocked):
    X, y = blocked[["x1", "x2"]], blocked["y"]
    splits = [(np.arange(30, len(blocked)), np.arange(30))]
    _, oof = cv.run_cross_validation(LinearRegression(), X, y, splits, inverse_transform=None)
    assert np.isnan(oof[30:]).all() and not np.isnan(oof[:30]).any()


def test_run_cross_validation_rejects_overlapping_splits(blocked):
    X, y = blocked[["x1", "x2"]], blocked["y"]
    with pytest.raises(ValueError):
        cv.run_cross_validation(LinearRegression(), X, y, [(np.arange(10), np.arange(5, 15))])
    with pytest.raises(ValueError):
        cv.run_cross_validation(LinearRegression(), X.to_numpy(), y, [(np.arange(10), np.arange(10, 15))])


def test_nearest_distance_is_a_great_circle_distance():
    one_degree = 2 * np.pi * cv.EARTH_RADIUS_KM / 360
    d = cv.nearest_distance_km([0.0], [0.0], [0.0, 10.0], [1.0, 0.0])
    assert d[0] == pytest.approx(one_degree, rel=1e-6)
    # a degree of longitude shrinks with latitude
    d60 = cv.nearest_distance_km([0.0], [60.0], [1.0], [60.0])
    assert d60[0] == pytest.approx(one_degree / 2, rel=2e-3)
    with pytest.raises(ValueError):
        cv.nearest_distance_km([200.0], [0.0], [0.0], [0.0])


def test_heldout_distances_are_zero_for_random_row_splits_and_positive_for_spatial(blocked):
    random_splits = list(KFold(5, shuffle=True, random_state=0).split(blocked))
    spatial_splits = cv.group_k_fold(blocked, 5)
    assert np.median(cv.heldout_distances_km(blocked, random_splits)) == 0
    assert cv.heldout_distances_km(blocked, spatial_splits).min() > 0


def test_summarize_distances_reports_percentiles_and_closest():
    table = cv.summarize_distances({"a": np.arange(1.0, 101.0)}, percentiles=(50,))
    assert table.loc["a", "50th percentile (km)"] == pytest.approx(50.5)
    assert table.loc["a", "closest (km)"] == 1.0
    with pytest.raises(ValueError):
        cv.summarize_distances({"a": [1.0, -2.0]})


def test_pooled_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert cv.pooled_metrics(y, y) == pytest.approx({"r2": 1.0, "rmse": 0.0, "mae": 0.0})
    with pytest.raises(ValueError):
        cv.pooled_metrics(y, np.array([1.0, np.nan, 3.0, 4.0]))


def test_paired_fold_comparison_matches_the_formula():
    a = pd.DataFrame({"fold": range(4), "rmse": [10.0, 12.0, 11.0, 13.0]})
    b = pd.DataFrame({"fold": range(4), "rmse": [11.0, 14.0, 11.5, 15.5]})
    out = cv.paired_fold_comparison(a, b)
    diff = a["rmse"] - b["rmse"]
    assert out["mean_diff"] == pytest.approx(diff.mean())
    assert out["t_stat"] == pytest.approx(diff.mean() / (diff.std(ddof=1) / 2))
    assert out["n_folds_a_better"] == 4


def test_station_interpolator_uses_nearest_station_of_the_same_month():
    month = pd.Period("2025-03", "M")
    train = pd.DataFrame({"longitude": [0.0, 10.0], "latitude": [0.0, 0.0], "month": [month, month]})
    y = pd.Series([1.0, 5.0])
    nearest = cv.StationInterpolator(n_neighbors=1).fit(train, y)
    test = pd.DataFrame({"longitude": [1.0, 9.0], "latitude": [0.0, 0.0], "month": [month, month]})
    assert nearest.predict(test).tolist() == [1.0, 5.0]
    # an exact location takes that station's value; an unseen month falls back to the training mean
    exact = pd.DataFrame({"longitude": [10.0], "latitude": [0.0], "month": [month]})
    assert nearest.predict(exact)[0] == 5.0
    other = pd.DataFrame({"longitude": [1.0], "latitude": [0.0], "month": [pd.Period("2025-04", "M")]})
    assert nearest.predict(other)[0] == pytest.approx(3.0)


def test_station_interpolator_weights_by_inverse_squared_distance_and_clones():
    month = pd.Period("2025-03", "M")
    train = pd.DataFrame({"longitude": [0.0, 3.0], "latitude": [0.0, 0.0], "month": [month, month]})
    model = cv.StationInterpolator(n_neighbors=2, power=2.0).fit(train, pd.Series([0.0, 12.0]))
    point = pd.DataFrame({"longitude": [1.0], "latitude": [0.0], "month": [month]})
    # distances ~1 and ~2 degrees: weights 1 and 1/4 -> (0*1 + 12*0.25) / 1.25 = 2.4 (great-circle ~ planar here)
    assert model.predict(point)[0] == pytest.approx(2.4, rel=1e-3)
    assert clone(model).get_params() == model.get_params()
    with pytest.raises(NotFittedError):
        cv.StationInterpolator().predict(point)


def test_permutation_importance_finds_the_informative_feature(blocked):
    X, y = blocked[["x1", "x2"]], blocked["y"]
    result = cv.permutation_importance_by_fold(LinearRegression(), X, y, cv.group_k_fold(blocked, 5), n_repeats=3, random_state=1)
    assert result["x1"].mean() > 10 * abs(result["x2"].mean())
    again = cv.permutation_importance_by_fold(LinearRegression(), X, y, cv.group_k_fold(blocked, 5), n_repeats=3, random_state=1)
    pd.testing.assert_frame_equal(result, again)
