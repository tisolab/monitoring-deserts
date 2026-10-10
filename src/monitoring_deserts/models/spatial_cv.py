"""Spatial cross-validation folds for the station-month table (project plan, Phase H).

Scoring a PM2.5 model on randomly held-out rows overstates how well it predicts where
there is no station, because a held-out station almost always has a training station a
few kilometres away. This module builds folds that hold out whole regions instead. It
handles four things:

1. clustering the stations into compact spatial blocks (KMeans on longitude/latitude);
2. giving every row of the station-month table the block of its station;
3. packing whole blocks into folds with scikit-learn's `GroupKFold`;
4. scoring a model over any set of folds, spatial or random, collecting per-fold metrics
   and out-of-fold predictions.

Blocks are fit on unique stations, not on station-month rows, so a station with a long
record doesn't pull the clusters toward itself, and every month of a station lands in
the same block and therefore in the same fold. The number of blocks and the number of
folds are separate choices: many small blocks are packed into a few folds. Why that
works, and how far apart the held-out and training stations end up on this project's
station network, is in the project plan (Phase H and Appendix D.8).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold

BLOCK_COLUMN = "spatial_block_id"


def _is_positive_int(value):
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        and value >= 1
    )


def cluster_stations(df, stations_per_block=10, n_init=15, random_state=42):
    """Cluster the stations of a station-month table into spatial blocks.

    Runs KMeans on the longitude/latitude of the *unique* stations, with
    `ceil(n_stations / stations_per_block)` clusters, and gives every row of `df` the
    block of its station. All months of a station therefore share one block, and so one
    fold.

    Block sizes are only roughly `stations_per_block`: stations concentrate in cities, so
    a few dense blocks come out much larger than the rest (project plan, Appendix D.8,
    explains why that is acceptable once the blocks are packed into folds). KMeans works
    on raw degrees, as in the project plan; at this region's latitudes a degree of
    longitude is up to about 20% shorter than a degree of latitude, which stretches the
    blocks slightly east-west.

    The blocks depend on the row order of `df`: KMeans seeds its clusters from the points
    in the order it receives them, so the same stations in a different row order can give
    a different partition (on this project's stations, an adjusted Rand index of 0.83-0.95
    against the saved order, not just renumbered labels). Keep the table's row order fixed
    if the folds must be reproducible.

    Parameters:
        df (pd.DataFrame): Station-month table with `station_id`, `longitude` and
            `latitude` columns. Each station must have the same coordinates in every row.
        stations_per_block (int): Target number of stations per block; sets the number of
            blocks. Defaults to 10.
        n_init (int): Number of KMeans restarts, keeping the best. Defaults to 15.
        random_state (int | None): KMeans seed, for reproducible blocks. Defaults to 42.

    Returns:
        pd.DataFrame: A copy of `df` with an integer `spatial_block_id` column, numbered
            0 to n_blocks - 1 (an existing column of that name is replaced). `df` itself
            is not modified.

    Raises:
        ValueError: If a required column is missing or `df` is empty; `station_id` or a
            coordinate is null, or a coordinate is out of range; a station has more than
            one coordinate pair; `stations_per_block` or `n_init` isn't a positive
            integer; the settings give fewer than 2 blocks; or there are fewer distinct
            station locations than blocks.
    """
    missing = {"station_id", "longitude", "latitude"} - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {sorted(missing)}")
    if df.empty:
        raise ValueError("df has no rows -- nothing to cluster.")
    if not _is_positive_int(stations_per_block):
        raise ValueError(
            f"stations_per_block must be a positive integer, got {stations_per_block!r}"
        )
    if not _is_positive_int(n_init):
        raise ValueError(f"n_init must be a positive integer, got {n_init!r}")

    null_columns = [
        c for c in ("station_id", "longitude", "latitude") if df[c].isna().any()
    ]
    if null_columns:
        raise ValueError(f"df has null values in column(s): {null_columns}")
    out_of_range = ~df["longitude"].between(-180, 180) | ~df["latitude"].between(
        -90, 90
    )
    if out_of_range.any():
        raise ValueError(
            f"{int(out_of_range.sum())} row(s) have a longitude/latitude out of range."
        )

    pairs_per_station = (
        df.drop_duplicates(["station_id", "longitude", "latitude"])
        .groupby("station_id")
        .size()
    )
    moving = pairs_per_station[pairs_per_station > 1].index.tolist()
    if moving:
        shown = moving[:5] + (["..."] if len(moving) > 5 else [])
        raise ValueError(
            f"{len(moving)} station(s) have more than one coordinate pair: {shown}"
        )

    new_df = df.copy()
    stations = new_df.drop_duplicates("station_id")[
        ["station_id", "longitude", "latitude"]
    ].copy()
    n_clusters = int(np.ceil(len(stations) / stations_per_block))
    if n_clusters < 2:
        raise ValueError(
            f"{len(stations)} stations at stations_per_block={stations_per_block} gives "
            f"{n_clusters} block; spatial cross-validation needs at least 2."
        )
    n_locations = len(stations[["longitude", "latitude"]].drop_duplicates())
    if n_locations < n_clusters:
        raise ValueError(
            f"only {n_locations} distinct station location(s) for {n_clusters} blocks; "
            "use a larger stations_per_block."
        )

    stations[BLOCK_COLUMN] = KMeans(
        n_clusters=n_clusters, n_init=n_init, random_state=random_state
    ).fit_predict(stations[["longitude", "latitude"]].to_numpy())
    new_df[BLOCK_COLUMN] = new_df["station_id"].map(
        stations.set_index("station_id")[BLOCK_COLUMN]
    )
    return new_df


def group_k_fold(df, n_folds=8):
    """Split a station-month table into folds that never separate a spatial block.

    Packs whole blocks into `n_folds` folds with scikit-learn's `GroupKFold`: a block is
    never split between train and test, and the folds come out close to equal in size
    (rows). With `n_folds` equal to the number of blocks this is leave-one-block-out.

    Parameters:
        df (pd.DataFrame): Station-month table with a `spatial_block_id` column, e.g. the
            output of `cluster_stations`.
        n_folds (int): Number of folds: at least 2, and at most the number of distinct
            blocks. Defaults to 8.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: One `(train_idx, test_idx)` pair per fold, as
            integer *positions* into `df` (use `.iloc`, not `.loc`). This is the format
            `GroupKFold.split` yields, so it feeds `cross_val_score`,
            `RandomizedSearchCV(cv=...)` and similar tools directly.

    Raises:
        ValueError: If `df` is empty, has no `spatial_block_id` column or has nulls in
            it, `n_folds` isn't an integer of at least 2, or there are fewer distinct
            blocks than folds.
    """
    if df.empty:
        raise ValueError("df has no rows -- nothing to split.")
    if BLOCK_COLUMN not in df.columns:
        raise ValueError(
            f"df has no '{BLOCK_COLUMN}' column; run cluster_stations first."
        )
    if df[BLOCK_COLUMN].isna().any():
        raise ValueError(
            f"'{BLOCK_COLUMN}' contains null values; every row needs a block."
        )
    if not _is_positive_int(n_folds) or n_folds < 2:
        raise ValueError(f"n_folds must be an integer of at least 2, got {n_folds!r}")
    n_blocks = df[BLOCK_COLUMN].nunique()
    if n_blocks < n_folds:
        raise ValueError(
            f"{n_blocks} distinct block(s) can't be packed into {n_folds} folds."
        )

    gkf = GroupKFold(n_splits=n_folds)
    # df only supplies the row count; the folds are computed from the blocks alone
    return list(gkf.split(df, groups=df[BLOCK_COLUMN]))


def run_cross_validation(model, X, y, splits, inverse_transform=np.expm1):
    """Score a model over a set of folds: per-fold metrics and out-of-fold predictions.

    For each `(train_idx, test_idx)` pair in `splits`, fits a fresh clone of `model` on
    the training rows, predicts the held-out rows and scores them. Any split source in
    the `GroupKFold.split` format works (the spatial folds from `group_k_fold`, a random
    `KFold`, ...), so the schemes can be compared with exactly the same model and metrics.

    The target is expected to be log-transformed. Predictions and actual values are
    passed through `inverse_transform` before scoring, so the metrics and the
    out-of-fold predictions are in the original units (ug/m3 for PM2.5). The default,
    `np.expm1`, undoes `log1p`; for a `log` target pass `np.exp`, and pass `None` to
    score on the model's own scale.

    Parameters:
        model: An unfitted scikit-learn-style estimator, e.g. a `RandomForestRegressor`.
            It is never fitted itself: each fold fits a clone, so the folds are
            independent.
        X (pd.DataFrame): Features, one row per station-month.
        y (pd.Series): Log-transformed target for the same rows (same index as `X`).
        splits (iterable): `(train_idx, test_idx)` pairs of integer row *positions* (not
            index labels), e.g. from `group_k_fold` or `KFold.split`. A list or a
            generator; it is read once.
        inverse_transform (callable | None): Applied to predictions and actual values
            before scoring. Defaults to `np.expm1`.

    Returns:
        tuple: `(metrics, oof_prediction)`.
            metrics (pd.DataFrame): One row per fold, with columns `fold` (counting from
                0), `n_test`, `r2`, `rmse` and `mae`.
            oof_prediction (np.ndarray): Out-of-fold predictions, one per row of `y` in
                the same order, after `inverse_transform`. `NaN` for a row that is in no
                test fold.

    Raises:
        ValueError: If `X` isn't a DataFrame or `y` a Series, they are empty, differ in
            length or index, or `y` has nulls; `inverse_transform` isn't callable or
            None; `splits` is empty or a split
            isn't a pair of non-empty integer arrays, has a test or train position out of
            range, or has a row in both train and test.
        RuntimeError: If fitting, predicting or scoring fails on a fold. The message names
            the fold (counting from 0) and the original exception is chained.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"X must be a pandas DataFrame, got {type(X).__name__}")
    if not isinstance(y, pd.Series):
        raise ValueError(f"y must be a pandas Series, got {type(y).__name__}")
    if len(X) == 0:
        raise ValueError("X has no rows -- nothing to cross-validate.")
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}.")
    if not X.index.equals(y.index):
        raise ValueError("X and y have different indexes; pass rows that line up.")
    if y.isna().any():
        raise ValueError(f"y has {int(y.isna().sum())} null value(s); drop those rows first.")
    if inverse_transform is not None and not callable(inverse_transform):
        raise ValueError("inverse_transform must be a callable or None.")

    n_rows = len(y)

    splits = list(splits)
    if not splits:
        raise ValueError("splits is empty -- nothing to cross-validate.")
    checked_splits = []
    for fold_n, split in enumerate(splits):
        try:
            train_idx, test_idx = split
        except (TypeError, ValueError):
            raise ValueError(f"split {fold_n} isn't a (train_idx, test_idx) pair") from None
        train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
        for name, idx in (("train", train_idx), ("test", test_idx)):
            if idx.ndim != 1 or idx.size == 0 or idx.dtype.kind not in "iu":
                raise ValueError(
                    f"fold {fold_n}: {name}_idx must be a non-empty 1-D array of integer positions"
                )
            if idx.min() < 0 or idx.max() >= n_rows:
                raise ValueError(
                    f"fold {fold_n}: {name}_idx has positions outside 0..{n_rows - 1}"
                )
        n_shared = np.intersect1d(train_idx, test_idx).size
        if n_shared:
            raise ValueError(
                f"fold {fold_n}: {n_shared} row(s) are in both the training and the test set"
            )
        checked_splits.append((train_idx, test_idx))

    n_folds = len(checked_splits)
    fold_results = []
    oof_prediction = np.full(n_rows, np.nan)

    for fold_n, (train_idx, test_idx) in enumerate(checked_splits):
        try:
            estimator = clone(model)
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = estimator.predict(X.iloc[test_idx])
            actual = y.iloc[test_idx].to_numpy()
            if inverse_transform is not None:
                pred, actual = inverse_transform(pred), inverse_transform(actual)
            scores = {
                "r2": r2_score(actual, pred),
                "rmse": root_mean_squared_error(actual, pred),
                "mae": mean_absolute_error(actual, pred),
            }
        except Exception as exc:
            raise RuntimeError(
                f"Cross-validation failed on fold {fold_n} (of {n_folds}, counting from 0): {exc}"
            ) from exc

        fold_results.append({"fold": fold_n, "n_test": len(test_idx), **scores})
        oof_prediction[test_idx] = pred

    return pd.DataFrame(fold_results), oof_prediction
