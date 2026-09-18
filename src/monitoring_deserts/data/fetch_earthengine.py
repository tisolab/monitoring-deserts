import warnings

import pandas as pd

import ee
import geemap


def stations_to_feature_collection(df, radius_m=2000):
    """Build an Earth Engine FeatureCollection of buffered station points.

    One `ee.Feature` per row of `df`, not per unique station -- a station
    legitimately appears more than once if `df` is a row-per-month ground
    truth table (as `03_satellite_features_extraction.ipynb` passes it).
    Each feature carries a `station_id` property, which a caller uses to
    match each `.reduceRegions()` result back to the row it came from. When
    `df` has more than one row per station (e.g. one per month), run
    `.reduceRegions()` separately per month window and match results on
    `station_id` within that single call -- `station_id` alone doesn't
    disambiguate which month a given feature's row belongs to.

    Parameters:
        df (pd.DataFrame): Must have `station_id`, `longitude`, `latitude`
            columns. `longitude`/`latitude` are plain WGS84 degrees.
        radius_m (int): Buffer radius, in meters, around each station
            point. Defaults to 2000 (2 km).

    Returns:
        ee.FeatureCollection: One feature per row of `df`, geometry a
            circular buffer around (longitude, latitude), with a
            `station_id` property.

    Raises:
        ValueError: If `df` is missing a required column, has no rows, or
            has a null or out-of-range longitude/latitude.
        RuntimeError: If Earth Engine rejects the request -- most commonly
            because `ee.Initialize(project=...)` wasn't called first.
    """
    required_cols = {"station_id", "longitude", "latitude"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {sorted(missing)}")
    if df.empty:
        raise ValueError("df has no rows -- nothing to build a FeatureCollection from.")

    bad_coords = (
        df["longitude"].isna()
        | df["latitude"].isna()
        | ~df["longitude"].between(-180, 180)
        | ~df["latitude"].between(-90, 90)
    )
    if bad_coords.any():
        raise ValueError(
            f"{int(bad_coords.sum())} row(s) have a missing or out-of-range "
            f"longitude/latitude: station_id(s) "
            f"{df.loc[bad_coords, 'station_id'].tolist()}"
        )

    try:
        features = [
            ee.Feature(
                ee.Geometry.Point([row.longitude, row.latitude]).buffer(radius_m),
                {"station_id": row.station_id},
            )
            for row in df.itertuples()
        ]
        return ee.FeatureCollection(features)
    except ee.EEException as exc:
        raise RuntimeError(
            f"Earth Engine rejected the FeatureCollection request: {exc}. "
            "Check that ee.Initialize(project=...) was called successfully "
            "before calling this function."
        ) from exc


def build_month_windows(DAY_START, DAY_END):
    """Build the list of monthly (start, end) windows for a regions reduction.

    Anchors every window boundary to DAY_START's own day-of-month (e.g. if
    DAY_START is the 15th, windows run 15th-to-15th), stepping forward one
    calendar month at a time up to DAY_END. This is deliberate rather than
    snapping to the 1st of each month: `pd.date_range(..., freq="MS")`
    silently drops the entire partial month between DAY_START and the next
    month-start whenever DAY_START isn't itself the 1st -- e.g.
    DAY_START="2025-03-15" would jump straight to "2025-04-01", silently
    dropping about two weeks of real study-period coverage with no warning
    at all.

    Parameters:
        DAY_START (str | pd.Timestamp | datetime.datetime): Start of the
            study period.
        DAY_END (str | pd.Timestamp | datetime.datetime): End of the study
            period.

    Returns:
        list[tuple[pd.Timestamp, pd.Timestamp]]: Consecutive
            (window_start, window_end) pairs, each one calendar month
            apart, covering DAY_START to DAY_END.

    Raises:
        ValueError: If DAY_START/DAY_END can't be parsed as timestamps, or
            if DAY_START is not before DAY_END.
    """
    try:
        start_ts = pd.Timestamp(DAY_START)
        end_ts = pd.Timestamp(DAY_END)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not parse DAY_START={DAY_START!r} / DAY_END={DAY_END!r} "
            f"as dates: {exc}"
        ) from exc

    if start_ts >= end_ts:
        raise ValueError(f"DAY_START ({start_ts}) must be before DAY_END ({end_ts}).")

    if start_ts.day != end_ts.day:
        warnings.warn(
            f"DAY_START ({start_ts.date()}) and DAY_END ({end_ts.date()}) fall "
            f"on different days of the month. Falling back to DAY_START's day "
            f"of month ({start_ts.day}) for every window boundary, including "
            f"the last one near DAY_END -- the final window may therefore end "
            f"a few days before or after the literal DAY_END you passed."
        )

    d_range = pd.date_range(start_ts, end_ts, freq=pd.DateOffset(months=1))
    return list(zip(d_range[:-1], d_range[1:]))


def pull_monthly_stats(
    image_collection, feature_collection, month_windows, im_name, scale=1000, on_month_done=None
):
    """Pull monthly-averaged Earth Engine band statistics over a FeatureCollection.

    For each (month_start, month_end) window in `month_windows`, takes the mean of
    `image_collection` over that window, reduces it over `feature_collection` (mean and
    standard deviation per feature, via `ee.Reducer.mean().combine(ee.Reducer.stdDev(), ...)`),
    and converts the result to a pandas DataFrame with a `month` column set to that window's
    start. One Earth Engine `reduceRegions` call per window, since `reduceRegions` only reduces
    over a single, already-filtered image.

    `geemap.ee_to_df(..., remove_geom=False)` plus a manual `.drop(columns=["geo"])` is used
    deliberately instead of `ee_to_df`'s default geometry removal, which hits a documented
    geemap/pandas incompatibility (`ValueError: Cannot specify both 'axis' and
    'index'/'columns'`) on current pandas versions. Don't "simplify" this back to the default.

    Parameters:
        image_collection (ee.ImageCollection): Band(s) to reduce, e.g. TROPOMI NO2 or AAI,
            already `.select()`-ed down to the band(s) of interest.
        feature_collection (ee.FeatureCollection): Regions to reduce over, e.g. from
            `stations_to_feature_collection`.
        month_windows (list[tuple]): (month_start, month_end) pairs, e.g. from
            `build_month_windows`.
        scale (int): Nominal reduction scale in meters. Defaults to 1000.
        on_month_done (callable | None): Optional callback invoked as
            `on_month_done(month_start, df)` immediately after each month's DataFrame is
            built, before it's appended to the returned list -- lets a caller persist
            progress incrementally (e.g. one file per month) so a failure partway through
            doesn't lose months already pulled. This function does no resuming itself: to
            resume after a partial failure, the caller should filter `month_windows` down to
            months not yet persisted before calling again -- the same pattern
            `pull_sensor_period` (`fetch_openaq.py`) uses via its own `on_sensor_done`.

    Returns:
        list[pd.DataFrame]: One DataFrame per month window, each with a `month` column plus
            whatever per-feature columns `ee_to_df` produced (station_id, band mean/stdDev,
            ...).

    Raises:
        ValueError: If `month_windows` is empty.
        RuntimeError: If Earth Engine rejects a given month's request -- includes which
            month failed and how many months succeeded before it.
    """
    if not month_windows:
        raise ValueError("month_windows is empty -- nothing to pull.")

    monthly_stats = []
    for month_start, month_end in month_windows:
        try:
            monthly_poll = image_collection.filterDate(month_start, month_end).mean()
            stats_feature_collection = monthly_poll.reduceRegions(
                collection=feature_collection,
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                scale=scale,
            )
            df = geemap.ee_to_df(stats_feature_collection, remove_geom=False).drop(
                columns=["geo"], errors="ignore"
            )
        except ee.EEException as exc:
            raise RuntimeError(
                f"Earth Engine rejected the request for month {month_start} "
                f"({len(monthly_stats)} of {len(month_windows)} months already "
                f"pulled): {exc}"
            ) from exc

        df["month"] = month_start
        if on_month_done is not None:
            on_month_done(month_start, df, im_name)
        monthly_stats.append(df)

    return monthly_stats
