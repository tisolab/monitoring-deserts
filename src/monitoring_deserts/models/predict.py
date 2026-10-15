"""Helpers for predicting PM2.5 on the analysis grid (project plan, Phase L).

The grid has about 435,000 cells and 16 months, close to 7 million rows, so
prediction needs a little more care than `model.predict(X)` on a few thousand
station-months. This module handles four things:

1. predicting in chunks, so a forest does not build one huge intermediate array;
2. turning log1p predictions back into micrograms per cubic metre, counting how many
   fall below zero (a linear model can return them) and clipping those at zero;
3. an uncertainty band from the out-of-fold residuals of the cross-validation: how far
   observed values were from the model's prediction at held-out stations, as quantiles;
4. flagging rows whose features fall outside what the training stations cover, where a
   linear model extrapolates and the prediction deserves less trust.

None of the uncertainty measures is a calibrated prediction interval. The residual band
is one global width taken from the stations, so it cannot know that a region is
unusual; the feature-range count is a warning, not a probability.
"""

import numpy as np
import pandas as pd


def predict_in_chunks(model, X, chunk_size=500_000):
    """Predict with `model` over the rows of `X`, `chunk_size` rows at a time.

    Parameters:
        model: A fitted estimator with a `predict` method (e.g. a scikit-learn
            pipeline or forest).
        X (pd.DataFrame): Feature table, one row per prediction. Must not contain
            nulls: a prediction for an incomplete row would be silently wrong or an
            error deep inside the model, so the caller selects complete rows first.
        chunk_size (int): Rows per call to `model.predict`. Defaults to 500,000.

    Returns:
        numpy.ndarray: 1-D float array of predictions, in the row order of `X`.

    Raises:
        ValueError: If `X` is not a non-empty DataFrame, contains nulls, or
            `chunk_size` is not a positive integer.
    """
    if not isinstance(X, pd.DataFrame) or len(X) == 0:
        raise ValueError("X must be a non-empty DataFrame.")
    if not (isinstance(chunk_size, int) and chunk_size > 0):
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    if X.isna().any().any():
        raise ValueError("X contains nulls; select the rows with every feature present first.")

    parts = [
        np.asarray(model.predict(X.iloc[start : start + chunk_size]), dtype=float).ravel()
        for start in range(0, len(X), chunk_size)
    ]
    return np.concatenate(parts)


def to_ugm3(log1p_prediction):
    """Back-transform log1p predictions to micrograms per cubic metre, clipped at zero.

    `np.expm1` of a log1p prediction below zero is negative, which is not a
    concentration. Those values are set to zero and counted, so the caller can report
    how many predictions were clipped.

    Parameters:
        log1p_prediction (array-like): Predictions on the log1p scale.

    Returns:
        tuple[numpy.ndarray, int]: `(values, n_clipped)`, the non-negative
            predictions in µg/m³ and how many were negative before clipping.
            NaN stays NaN and is not counted.

    Raises:
        ValueError: If the input is empty.
    """
    pred = np.asarray(log1p_prediction, dtype=float)
    if pred.size == 0:
        raise ValueError("log1p_prediction is empty.")
    values = np.expm1(pred)
    n_clipped = int((values < 0).sum())
    return np.where(values < 0, 0.0, values), n_clipped


def residual_offsets(y_true, y_pred, level=0.9):
    """Lower and upper quantile offsets of the prediction residuals.

    The residual is `y_true - y_pred`. Adding the two offsets to a prediction gives a
    band that contained `level` of the residuals seen at the stations that were held
    out of training. Use it on the scale the model was fit on (log1p here).

    Parameters:
        y_true (array-like): Observed values at held-out stations.
        y_pred (array-like): Out-of-fold predictions, same length and scale.
        level (float): Share of residuals the band should contain, strictly between
            0 and 1. Defaults to 0.9.

    Returns:
        tuple[float, float]: `(lower_offset, upper_offset)`, the `(1-level)/2` and
            `(1+level)/2` quantiles of the residuals. The lower offset is usually
            negative.

    Raises:
        ValueError: If the inputs differ in length, have fewer than 20 values or
            contain non-finite values, or `level` is not strictly between 0 and 1.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be 1-D arrays of the same length.")
    if y_true.size < 20:
        raise ValueError(f"Need at least 20 residuals to estimate quantiles, got {y_true.size}.")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("y_true and y_pred must be finite (out-of-fold predictions with NaN mean untested rows).")
    if not 0 < level < 1:
        raise ValueError(f"level must be strictly between 0 and 1, got {level!r}")

    lower, upper = np.quantile(y_true - y_pred, [(1 - level) / 2, (1 + level) / 2])
    return float(lower), float(upper)


def count_features_outside_range(X, reference, lower_q=0.01, upper_q=0.99):
    """Count, per row, how many features lie outside the range of a reference table.

    The reference is the training stations' feature table. A feature is "outside"
    when its value is below the reference's `lower_q` quantile or above its `upper_q`
    quantile. A high count means the row is unlike the training data in several
    features at once, which is where a linear model extrapolates.

    Parameters:
        X (pd.DataFrame): Rows to check (e.g. the grid), numeric columns.
        reference (pd.DataFrame): Training features; must contain every column of `X`.
        lower_q (float): Lower quantile of the reference range. Defaults to 0.01.
        upper_q (float): Upper quantile of the reference range. Defaults to 0.99.

    Returns:
        tuple[numpy.ndarray, pd.DataFrame]: `(counts, ranges)`. `counts` is an integer
            array with one entry per row of `X`. `ranges` has one row per column of `X`
            with the `lower` and `upper` limits used. Missing values in `X` are not
            counted as outside.

    Raises:
        ValueError: If a column of `X` is absent from `reference`, either table is
            empty, or the quantiles are not `0 <= lower_q < upper_q <= 1`.
    """
    if len(X) == 0 or len(reference) == 0:
        raise ValueError("X and reference must be non-empty.")
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError(f"Need 0 <= lower_q < upper_q <= 1, got {lower_q!r}, {upper_q!r}")
    missing = [column for column in X.columns if column not in reference.columns]
    if missing:
        raise ValueError(f"Columns of X missing from the reference table: {missing}")

    ranges = pd.DataFrame(
        {
            "lower": reference[X.columns].quantile(lower_q),
            "upper": reference[X.columns].quantile(upper_q),
        }
    )
    outside = (X.lt(ranges["lower"], axis=1) | X.gt(ranges["upper"], axis=1)).sum(axis=1)
    return outside.to_numpy(dtype=int), ranges
