"""Spatial cross-validation folds for the station-month table (project plan, Phase H).

Scoring a PM2.5 model on randomly held-out rows overstates how well it predicts where
there is no station, because a held-out station almost always has a training station a
few kilometres away. This module builds folds that hold out whole regions instead. It
handles eight things:

1. clustering the stations into compact spatial blocks (KMeans on longitude/latitude);
2. giving every row of the station-month table the block of its station;
3. packing whole blocks into folds with scikit-learn's `GroupKFold`;
4. scoring a model over any set of folds, spatial or random, collecting per-fold metrics
   and out-of-fold predictions;
5. checking how realistic a fold scheme is: how far its held-out stations are from the
   nearest training station, against how far the cells to be predicted are from the
   nearest station (the distance check);
6. measuring which features a model relies on when it predicts held-out regions
   (permutation importance, fold by fold);
7. a baseline that uses no features at all, only the locations and values of the
   training stations (nearest-station and inverse-distance interpolation), to show what
   the features add;
8. summarising scores: pooled out-of-fold metrics, and a paired comparison of two
   models fold by fold.

Blocks are fit on unique stations, not on station-month rows, so a station with a long
record doesn't pull the clusters toward itself, and every month of a station lands in
the same block and therefore in the same fold. The number of blocks and the number of
folds are separate choices: many small blocks are packed into a few folds. Why that
works, and how far apart the held-out and training stations end up on this project's
station network, is in the project plan (Phase H and Appendix D.8).
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted

BLOCK_COLUMN = "spatial_block_id"
EARTH_RADIUS_KM = 6371.0088  # mean radius; great-circle distances treat the Earth as a sphere


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


def _check_splits(splits, n_rows):
    """Validate `(train_idx, test_idx)` pairs of row positions; return them as a list of arrays."""
    splits = list(splits)
    if not splits:
        raise ValueError("splits is empty -- nothing to evaluate.")
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
    return checked_splits


def _resolve_feature_groups(columns, feature_groups):
    """Turn `feature_groups` into `{name: [columns]}` covering every column exactly once.

    Columns that no group mentions become a group of their own, named after the column.
    """
    if feature_groups is None:
        return {column: [column] for column in columns}
    if not isinstance(feature_groups, dict):
        raise ValueError("feature_groups must be a dict {name: [columns]} or None.")

    groups, used = {}, set()
    for name, group_columns in feature_groups.items():
        if not isinstance(name, str):
            raise ValueError(f"feature group names must be strings, got {name!r}")
        if not isinstance(group_columns, (list, tuple)) or len(group_columns) == 0:
            raise ValueError(f"feature group {name!r} must be a non-empty list of columns.")
        unknown = [c for c in group_columns if c not in columns]
        if unknown:
            raise ValueError(f"feature group {name!r} has column(s) not in X: {unknown}")
        repeated = [c for c in group_columns if c in used or list(group_columns).count(c) > 1]
        if repeated:
            raise ValueError(
                f"column(s) {sorted(set(repeated))} appear in more than one feature group "
                f"(or twice in {name!r})."
            )
        used.update(group_columns)
        groups[name] = list(group_columns)

    for column in columns:
        if column not in used:
            if column in groups:
                raise ValueError(
                    f"feature group {column!r} has the same name as a column that is in no group."
                )
            groups[column] = [column]
    return groups


def _rmse_of(estimator, X_eval, actual, inverse_transform):
    """RMSE of `estimator` on `X_eval` against `actual` (already inverse-transformed)."""
    pred = estimator.predict(X_eval)
    if inverse_transform is not None:
        pred = inverse_transform(pred)
    return root_mean_squared_error(actual, pred)


def permutation_importance_by_fold(
    model, X, y, splits, feature_groups=None, n_repeats=10, random_state=42,
    inverse_transform=None,
):
    """Permutation importance of each feature (or feature group) on every held-out fold.

    For each `(train_idx, test_idx)` pair in `splits`, fits a fresh clone of `model` on the
    training rows and records its RMSE on the held-out rows. Then, one feature group at a
    time, it shuffles the group's values across the held-out rows, predicts again and
    records how much the RMSE grew. A large increase means the model relies on that
    feature to predict regions it was not trained on. Unlike a model's built-in
    importances, which are computed on training data, this measures what the feature
    contributes on data the model has not seen, which is what the spatial folds are for.

    The columns of a group are shuffled together with the same row permutation, so
    features that only make sense together (e.g. `sin_month` and `cos_month`) are
    broken as one. Correlated features that are not grouped can hide each other: the
    model recovers the shuffled one from its partners, so each looks less important than
    it is. The result describes the model, not a causal effect.

    Parameters:
        model: An unfitted scikit-learn-style estimator. It is never fitted itself: each
            fold fits a clone.
        X (pd.DataFrame): Features, one row per station-month. Categorical dtypes are
            kept when shuffling.
        y (pd.Series): Target for the same rows (same index as `X`).
        splits (iterable): `(train_idx, test_idx)` pairs of integer row *positions*, e.g.
            from `group_k_fold` or `KFold.split`. A list or a generator; it is read once.
        feature_groups (dict | None): `{group_name: [columns]}`. Columns that are in no
            group are scored on their own, under the column's name. `None` scores every
            column on its own.
        n_repeats (int): Shuffles per feature group and fold; the increases are averaged.
        random_state (int): Seed of the shuffles; the same seed gives the same result.
        inverse_transform (callable | None): Applied to the predictions and the actual
            values before computing the RMSE. The default `None` scores on the model's own
            scale (for a `log1p` target, the log scale the tuning used); pass `np.expm1`
            for ug/m3, where a few high-pollution folds dominate.

    Returns:
        pd.DataFrame: One row per fold (index `fold`, counting from 0) and one column per
            feature group: the mean increase in RMSE when that group is shuffled. Positive
            means the model relies on it; around zero means it doesn't; slightly negative
            values are noise.

    Raises:
        ValueError: If `X` isn't a DataFrame or `y` a Series, they are empty, differ in
            length or index, or `y` has nulls; `inverse_transform` isn't callable or None;
            `n_repeats` isn't a positive integer or `random_state` isn't an integer;
            `feature_groups` is malformed, has an unknown column, or puts a column in two
            groups; `splits` is empty or a split isn't a pair of non-empty integer arrays,
            has a position out of range, or has a row in both train and test.
        RuntimeError: If fitting or predicting fails on a fold. The message names the fold
            (counting from 0) and the original exception is chained.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"X must be a pandas DataFrame, got {type(X).__name__}")
    if not isinstance(y, pd.Series):
        raise ValueError(f"y must be a pandas Series, got {type(y).__name__}")
    if len(X) == 0:
        raise ValueError("X has no rows -- nothing to evaluate.")
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}.")
    if not X.index.equals(y.index):
        raise ValueError("X and y have different indexes; pass rows that line up.")
    if y.isna().any():
        raise ValueError(f"y has {int(y.isna().sum())} null value(s); drop those rows first.")
    if inverse_transform is not None and not callable(inverse_transform):
        raise ValueError("inverse_transform must be a callable or None.")
    if not _is_positive_int(n_repeats):
        raise ValueError(f"n_repeats must be a positive integer, got {n_repeats!r}")
    if not isinstance(random_state, (int, np.integer)) or isinstance(random_state, bool):
        raise ValueError(f"random_state must be an integer, got {random_state!r}")

    groups = _resolve_feature_groups(list(X.columns), feature_groups)
    checked_splits = _check_splits(splits, len(y))
    n_folds = len(checked_splits)
    rng = np.random.default_rng(random_state)

    rows = []
    for fold_n, (train_idx, test_idx) in enumerate(checked_splits):
        try:
            estimator = clone(model)
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            X_test = X.iloc[test_idx]
            actual = y.iloc[test_idx].to_numpy()
            if inverse_transform is not None:
                actual = inverse_transform(actual)
            baseline = _rmse_of(estimator, X_test, actual, inverse_transform)

            increases = {}
            for name, group_columns in groups.items():
                per_repeat = []
                for _ in range(n_repeats):
                    perm = rng.permutation(len(X_test))
                    X_shuffled = X_test.copy()
                    for column in group_columns:
                        # .array keeps the dtype (categoricals too) and ignores the index labels
                        X_shuffled[column] = X_test[column].iloc[perm].array
                    per_repeat.append(
                        _rmse_of(estimator, X_shuffled, actual, inverse_transform) - baseline
                    )
                increases[name] = float(np.mean(per_repeat))
        except Exception as exc:
            raise RuntimeError(
                f"Permutation importance failed on fold {fold_n} (of {n_folds}, counting from 0): {exc}"
            ) from exc
        rows.append({"fold": fold_n, **increases})

    return pd.DataFrame(rows).set_index("fold")


def _to_unit_sphere(longitude, latitude):
    """Longitude/latitude in degrees -> 3-D points on the unit sphere."""
    lon, lat = np.radians(longitude), np.radians(latitude)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _check_coordinates(longitude, latitude, label):
    """Return longitude and latitude as float arrays, after checking they are usable points."""
    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    if lon.ndim != 1 or lon.shape != lat.shape or lon.size == 0:
        raise ValueError(
            f"{label} longitude and latitude must be non-empty 1-D arrays of the same length."
        )
    if np.isnan(lon).any() or np.isnan(lat).any():
        raise ValueError(f"{label} coordinates contain null values.")
    if (np.abs(lon) > 180).any() or (np.abs(lat) > 90).any():
        raise ValueError(
            f"{label} coordinates are out of range (longitude within 180, latitude within 90)."
        )
    return lon, lat


def nearest_distance_km(from_longitude, from_latitude, to_longitude, to_latitude):
    """Great-circle distance from each "from" point to its nearest "to" point, in km.

    Points are converted to the unit sphere, where straight-line distance maps exactly to
    great-circle distance, and the nearest neighbour is found with a KD-tree. That avoids
    working in raw degrees (a degree of longitude shrinks with latitude) and, unlike an
    equal-area projection, keeps distances true. The Earth is treated as a sphere
    (`EARTH_RADIUS_KM`), which is typically within about 0.5% of ellipsoidal distances.

    Parameters:
        from_longitude, from_latitude (array-like): Coordinates in degrees of the points to
            measure from, e.g. grid-cell centres or held-out stations.
        to_longitude, to_latitude (array-like): Coordinates in degrees of the points to
            measure to, e.g. the stations.

    Returns:
        numpy.ndarray: One distance in km per "from" point, in the same order.

    Raises:
        ValueError: If either set of coordinates is empty, not 1-D, of mismatched length,
            contains nulls, or has a longitude beyond 180 or a latitude beyond 90 degrees.
    """
    from_lon, from_lat = _check_coordinates(from_longitude, from_latitude, "from")
    to_lon, to_lat = _check_coordinates(to_longitude, to_latitude, "to")

    chord, _ = cKDTree(_to_unit_sphere(to_lon, to_lat)).query(_to_unit_sphere(from_lon, from_lat))
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(chord, 2) / 2)  # chord length -> arc length


def heldout_distances_km(df, splits):
    """Distance from each held-out station to the nearest training station, pooled over folds.

    For each `(train_idx, test_idx)` pair, takes the distinct stations among the training
    rows and among the test rows (a station has many monthly rows) and measures from every
    held-out station to the closest training station. This is what a fold scheme rehearses:
    how far the model's test locations are from the stations it learned from, to compare
    with how far the cells to be predicted are from the nearest station.

    With a random split over rows, a held-out station's other months are usually in the
    training set, so its own location is a training location and its distance is 0.

    Parameters:
        df (pd.DataFrame): Station-month table with `station_id`, `longitude` and `latitude`
            columns. Each station must have the same coordinates in every row.
        splits (iterable): `(train_idx, test_idx)` pairs of integer row positions into `df`
            (not index labels), e.g. from `group_k_fold` or `KFold.split`. A list or a
            generator; it is read once.

    Returns:
        numpy.ndarray: One distance in km per held-out station per fold, the folds
            concatenated in order.

    Raises:
        ValueError: If a required column is missing or `df` is empty; `station_id`,
            longitude or latitude has nulls; a station has more than one coordinate pair;
            `splits` is empty, or a split isn't a pair of non-empty integer arrays, or has
            a position out of range. Out-of-range coordinates are reported by
            `nearest_distance_km`.
    """
    missing = {"station_id", "longitude", "latitude"} - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {sorted(missing)}")
    if df.empty:
        raise ValueError("df has no rows -- nothing to measure.")
    null_columns = [c for c in ("station_id", "longitude", "latitude") if df[c].isna().any()]
    if null_columns:
        raise ValueError(f"df has null values in column(s): {null_columns}")
    pairs_per_station = (
        df.drop_duplicates(["station_id", "longitude", "latitude"]).groupby("station_id").size()
    )
    moving = pairs_per_station[pairs_per_station > 1].index.tolist()
    if moving:
        shown = moving[:5] + (["..."] if len(moving) > 5 else [])
        raise ValueError(
            f"{len(moving)} station(s) have more than one coordinate pair: {shown}"
        )

    stations = df.drop_duplicates("station_id").set_index("station_id")[["longitude", "latitude"]]
    row_station_id = df["station_id"].to_numpy()
    n_rows = len(df)

    splits = list(splits)
    if not splits:
        raise ValueError("splits is empty -- nothing to measure.")

    pooled = []
    for fold_n, split in enumerate(splits):
        try:
            train_idx, test_idx = split
        except (TypeError, ValueError):
            raise ValueError(f"split {fold_n} isn't a (train_idx, test_idx) pair") from None
        fold_stations = {}
        for name, idx in (("train", train_idx), ("test", test_idx)):
            idx = np.asarray(idx)
            if idx.ndim != 1 or idx.size == 0 or idx.dtype.kind not in "iu":
                raise ValueError(
                    f"fold {fold_n}: {name}_idx must be a non-empty 1-D array of integer positions"
                )
            if idx.min() < 0 or idx.max() >= n_rows:
                raise ValueError(
                    f"fold {fold_n}: {name}_idx has positions outside 0..{n_rows - 1}"
                )
            fold_stations[name] = stations.loc[np.unique(row_station_id[idx])]
        train, test = fold_stations["train"], fold_stations["test"]
        pooled.append(
            nearest_distance_km(
                test["longitude"], test["latitude"], train["longitude"], train["latitude"]
            )
        )
    return np.concatenate(pooled)


def summarize_distances(distances_km, percentiles=(10, 25, 50, 75, 90)):
    """Summarise sets of distances by their percentiles, one row per set.

    Meant for putting the outputs of `nearest_distance_km` and `heldout_distances_km` side by
    side: the distances from the cells to be predicted to the nearest station against those
    from each fold scheme's held-out stations to the nearest training station.

    Parameters:
        distances_km (dict[str, array-like]): Label -> distances in km, e.g.
            `{"prediction cells": reality_km, "spatial folds": heldout_distances_km(df, splits)}`.
        percentiles (sequence of float): Percentiles to report, each between 0 and 100.
            Defaults to 10, 25, 50, 75 and 90.

    Returns:
        pd.DataFrame: One row per label, one column per percentile (named like
            "50th percentile (km)") and a final "closest (km)" column holding the smallest
            distance in the set. A dead zone (buffer) around the test stations only removes
            training stations if it is larger than the closest distance of a fold scheme.

    Raises:
        ValueError: If `distances_km` isn't a non-empty dict, `percentiles` is empty or has a
            value outside 0-100, or a set of distances is empty, not 1-D, negative or has
            nulls.
    """
    if not isinstance(distances_km, dict) or not distances_km:
        raise ValueError("distances_km must be a non-empty dict of label -> distances.")
    levels = np.asarray(percentiles, dtype=float)
    if levels.ndim != 1 or levels.size == 0 or (levels < 0).any() or (levels > 100).any():
        raise ValueError("percentiles must be a non-empty sequence of numbers between 0 and 100.")

    summary, closest = {}, []
    for label, values in distances_km.items():
        km = np.asarray(values, dtype=float)
        if km.ndim != 1 or km.size == 0:
            raise ValueError(f"'{label}': distances must be a non-empty 1-D array.")
        if np.isnan(km).any() or (km < 0).any():
            raise ValueError(f"'{label}': distances must be non-negative numbers without nulls.")
        summary[label] = np.percentile(km, levels)
        closest.append(km.min())

    table = pd.DataFrame(summary, index=[f"{p:g}th percentile (km)" for p in levels]).T
    table["closest (km)"] = closest
    return table


def pooled_metrics(y_obs, oof_prediction):
    """R2, RMSE and MAE of all out-of-fold predictions scored together, in one go.

    The headline numbers of a cross-validation: every held-out prediction is pooled and
    scored once, instead of averaging per-fold scores (which gives a small or unusual fold
    the same weight as a large one, and measures R2 against each fold's own mean). Use the
    observed values and the `oof_prediction` returned by `run_cross_validation`, both in
    the same (original) units.

    Parameters:
        y_obs (array-like): Observed values, one per row.
        oof_prediction (array-like): Out-of-fold predictions for the same rows, in the same
            order.

    Returns:
        dict: `{"r2": ..., "rmse": ..., "mae": ...}` as plain floats.

    Raises:
        ValueError: If either input is empty or not 1-D, they differ in length, or either
            has nulls (a `NaN` out-of-fold prediction is a row that was never held out).
    """
    observed = np.asarray(y_obs, dtype=float)
    predicted = np.asarray(oof_prediction, dtype=float)
    if observed.ndim != 1 or predicted.ndim != 1 or observed.size == 0:
        raise ValueError("y_obs and oof_prediction must be non-empty 1-D arrays.")
    if observed.shape != predicted.shape:
        raise ValueError(
            f"y_obs has {observed.size} values but oof_prediction has {predicted.size}."
        )
    if np.isnan(observed).any():
        raise ValueError(f"y_obs has {int(np.isnan(observed).sum())} null value(s).")
    if np.isnan(predicted).any():
        raise ValueError(
            f"oof_prediction has {int(np.isnan(predicted).sum())} null value(s): "
            "those rows were never in a test fold."
        )
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(root_mean_squared_error(observed, predicted)),
        "mae": float(mean_absolute_error(observed, predicted)),
    }


def paired_fold_comparison(metrics_a, metrics_b, column="rmse"):
    """Compare two models fold by fold with a paired test on a per-fold metric.

    Both models were scored on the same folds, so the difference on each fold removes how
    hard that fold is, and the spread of those differences (not the spread of the metric
    itself) is the right noise scale. With `diff = a - b` per fold, the statistic is the
    mean difference divided by its standard error, `std(diff) / sqrt(n_folds)`. For an
    error metric such as RMSE a negative value means model A is better. With few folds
    (this project has 8) and folds that are not fully independent, treat the statistic as
    a rough guide, around 2 in absolute value being borderline.

    Parameters:
        metrics_a, metrics_b (pd.DataFrame): Per-fold metrics as returned by
            `run_cross_validation`, with columns `fold` and `column`, scored on the same
            folds.
        column (str): The metric to compare, e.g. `"rmse"`.

    Returns:
        dict: `n_folds`, `mean_diff`, `std_diff` (sample standard deviation, ddof=1),
            `std_err`, `t_stat` and `n_folds_a_better` (folds where `a < b`). If the
            differences have no spread, `t_stat` is 0.0 when `mean_diff` is 0 and
            otherwise plus or minus infinity.

    Raises:
        ValueError: If an input isn't a DataFrame with the needed columns, has repeated or
            missing folds, the two fold sets differ, there are fewer than 2 folds, or the
            metric has nulls.
    """
    frames = {}
    for label, frame in (("metrics_a", metrics_a), ("metrics_b", metrics_b)):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{label} must be a pandas DataFrame, got {type(frame).__name__}")
        missing = {"fold", column} - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing column(s): {sorted(missing)}")
        if frame["fold"].duplicated().any():
            raise ValueError(f"{label} has repeated fold numbers.")
        if frame[column].isna().any():
            raise ValueError(f"{label} has null values in {column!r}.")
        frames[label] = frame.set_index("fold")[column].astype(float)
    if set(frames["metrics_a"].index) != set(frames["metrics_b"].index):
        raise ValueError("metrics_a and metrics_b were not scored on the same folds.")
    n_folds = len(frames["metrics_a"])
    if n_folds < 2:
        raise ValueError("a paired comparison needs at least 2 folds.")

    diff = frames["metrics_a"] - frames["metrics_b"].reindex(frames["metrics_a"].index)
    mean_diff = float(diff.mean())
    std_diff = float(diff.std())
    std_err = std_diff / np.sqrt(n_folds)
    if std_err > 0:
        t_stat = mean_diff / std_err
    else:
        t_stat = 0.0 if mean_diff == 0 else float(np.sign(mean_diff) * np.inf)
    return {
        "n_folds": int(n_folds),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "std_err": float(std_err),
        "t_stat": float(t_stat),
        "n_folds_a_better": int((diff < 0).sum()),
    }


class StationInterpolator(RegressorMixin, BaseEstimator):
    """Predict a held-out station-month from nearby training stations of the same month.

    A baseline that uses no features at all: the prediction for a row is the inverse-distance
    weighted average of the target at the `n_neighbors` nearest training stations that have a
    value in the same time period (great-circle distance). With `n_neighbors=1` it is the
    plain nearest-station prediction. If a model with satellite and weather features cannot
    clearly beat this, the features add little beyond interpolating the stations.

    It follows the scikit-learn estimator interface, so it can be passed to
    `run_cross_validation` unchanged; only the training rows given to `fit` are ever used,
    so nothing from the held-out stations leaks. The average is taken on the target's own
    scale (the log scale for a log-transformed target), consistent with how the other
    models are scored.

    Parameters:
        n_neighbors (int): Number of nearest training stations to average. With fewer
            training stations in the period, all of them are used.
        power (float): Exponent of the inverse-distance weights, `1 / distance**power`.
            A held-out row at exactly the location of a training row takes that value.
        longitude_column, latitude_column (str): Coordinate columns of `X`, in degrees.
        time_column (str): Column of `X` identifying the period (any hashable values, e.g.
            the year-month); only stations of the same period are used as neighbours.

    Attributes:
        mean_ (float): Mean of the training target, used for any period with no training
            rows (including periods not seen in `fit`).
    """

    def __init__(
        self, n_neighbors=5, power=2.0, longitude_column="longitude",
        latitude_column="latitude", time_column="month",
    ):
        self.n_neighbors = n_neighbors
        self.power = power
        self.longitude_column = longitude_column
        self.latitude_column = latitude_column
        self.time_column = time_column

    def _prepare(self, X):
        """Check `X`; return longitude, latitude and the period of every row."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a pandas DataFrame, got {type(X).__name__}")
        columns = [self.longitude_column, self.latitude_column, self.time_column]
        missing = [c for c in columns if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing required column(s): {missing}")
        if len(X) == 0:
            raise ValueError("X has no rows.")
        if X[columns].isna().any().any():
            raise ValueError("X has null values in the coordinate or time columns.")
        lon, lat = _check_coordinates(X[self.longitude_column], X[self.latitude_column], "X")
        return lon, lat, X[self.time_column].reset_index(drop=True)

    def fit(self, X, y):
        """Index the training stations of every period; `y` is the training target."""
        if not _is_positive_int(self.n_neighbors):
            raise ValueError(f"n_neighbors must be a positive integer, got {self.n_neighbors!r}")
        if (
            isinstance(self.power, bool)
            or not isinstance(self.power, (int, float, np.integer, np.floating))
            or not np.isfinite(self.power)
            or self.power <= 0
        ):
            raise ValueError(f"power must be a positive number, got {self.power!r}")
        lon, lat, period = self._prepare(X)
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or target.size != len(lon):
            raise ValueError(f"y must be 1-D with one value per row of X ({len(lon)}).")
        if np.isnan(target).any():
            raise ValueError("y has null values.")

        points = _to_unit_sphere(lon, lat)
        self.trees_ = {}
        for value in pd.unique(period):
            rows = (period == value).to_numpy()
            self.trees_[value] = (cKDTree(points[rows]), target[rows])
        self.mean_ = float(target.mean())
        return self

    def predict(self, X):
        """Interpolate the target for every row of `X` from training stations of its period."""
        check_is_fitted(self, "trees_")
        lon, lat, period = self._prepare(X)
        points = _to_unit_sphere(lon, lat)
        prediction = np.full(len(lon), self.mean_)

        for value in pd.unique(period):
            if value not in self.trees_:
                continue                                  # unseen period: the training mean
            rows = (period == value).to_numpy()
            tree, values = self.trees_[value]
            k = min(self.n_neighbors, len(values))
            chord, neighbour = tree.query(points[rows], k=k)
            chord, neighbour = chord.reshape(-1, k), neighbour.reshape(-1, k)
            km = 2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(chord, 2) / 2)
            neighbour_values = values[neighbour]

            exact = km <= 1e-6                            # at the location of a training row
            safe_km = np.where(exact, 1.0, km)
            weights = np.where(
                exact.any(axis=1, keepdims=True), exact.astype(float), 1.0 / safe_km ** self.power
            )
            prediction[rows] = (weights * neighbour_values).sum(axis=1) / weights.sum(axis=1)
        return prediction
