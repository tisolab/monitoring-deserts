import warnings

import pandas as pd

import ee


def stations_to_feature_collection(df, radius_m=2000):
    """Build an Earth Engine FeatureCollection of buffered station points.

    One `ee.Feature` per row of `df`, not per unique station -- a station
    legitimately appears more than once if `df` is a row-per-month ground
    truth table (as `03_satellite_features_extraction.ipynb` passes it).
    Each feature carries a `station_id` property so a downstream
    `.reduceRegions()` result can be joined back to the right row.

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
    at all. Don't "simplify" this back to `freq="MS"`.

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
