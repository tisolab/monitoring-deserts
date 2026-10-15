import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from monitoring_deserts.models import predict


@pytest.fixture
def features():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1000, 3)), columns=list("abc"))
    return X, 2 * X["a"] + rng.normal(0, 0.1, 1000)


def test_predict_in_chunks_equals_a_single_call(features):
    X, y = features
    model = LinearRegression().fit(X, y)
    assert np.allclose(predict.predict_in_chunks(model, X, chunk_size=300), model.predict(X))
    assert predict.predict_in_chunks(model, X, chunk_size=5000).shape == (1000,)


def test_predict_in_chunks_rejects_nulls_and_bad_chunks(features):
    X, y = features
    model = LinearRegression().fit(X, y)
    with_null = X.copy()
    with_null.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        predict.predict_in_chunks(model, with_null)
    with pytest.raises(ValueError):
        predict.predict_in_chunks(model, X, chunk_size=0)
    with pytest.raises(ValueError):
        predict.predict_in_chunks(model, X.iloc[:0])


def test_to_ugm3_clips_negatives_and_counts_them():
    values, n_clipped = predict.to_ugm3([np.log1p(10), -0.5, np.nan, 0.0])
    assert values[0] == pytest.approx(10) and values[1] == 0 and values[3] == 0 and np.isnan(values[2])
    assert n_clipped == 1
    with pytest.raises(ValueError):
        predict.to_ugm3([])


def test_residual_offsets_are_the_residual_quantiles():
    lower, upper = predict.residual_offsets(np.arange(101.0), np.zeros(101), level=0.9)
    assert (lower, upper) == pytest.approx((5.0, 95.0))
    with pytest.raises(ValueError):
        predict.residual_offsets(np.ones(30), np.ones(30), level=1)
    with pytest.raises(ValueError):
        predict.residual_offsets(np.ones(5), np.ones(5))  # too few residuals
    with pytest.raises(ValueError):
        predict.residual_offsets(np.r_[np.ones(29), np.nan], np.ones(30))


def test_count_features_outside_range(features):
    X, _ = features
    rows = pd.DataFrame({"a": [0, 100, np.nan], "b": [0, 100, 0], "c": [0, 0, 0]})
    counts, ranges = predict.count_features_outside_range(rows, X)
    assert counts.tolist() == [0, 2, 0]  # a missing value is not counted as outside
    assert ranges.shape == (3, 2)
    with pytest.raises(ValueError):
        predict.count_features_outside_range(rows.assign(z=1), X)
    with pytest.raises(ValueError):
        predict.count_features_outside_range(rows, X, lower_q=0.9, upper_q=0.1)
