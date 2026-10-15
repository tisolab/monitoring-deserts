import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import rasterio.errors
import requests
from rasterio.io import MemoryFile
from tqdm.auto import tqdm

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


def download_image_on_grid(
    image,
    grid_lons,
    grid_lats,
    cell_size,
    tile_cells=200,
    max_source_pixels=1024,
    progress=True,
    reducer=None,
    bands=None,
    max_workers=1,
    retries=0,
):
    """Download an Earth Engine image reduced onto a regular grid, as a numpy array.

    For grid-cell features where `reduceRegions` doesn't scale (hundreds of
    thousands of cells) but the *output* is small: each cell's value is the
    mean of the source pixels inside it (`reduceResolution` with a mean
    reducer by default, then a reprojection onto the grid), and the result comes back as
    small GeoTIFF downloads, one per tile, with no Drive export or waiting on
    an asynchronous task. Tiling keeps each request light enough not to hit
    Earth Engine's interactive compute and size limits.

    Parameters:
        image (ee.Image): An image with a fixed projection (e.g.
            `collection.first().select("band")`); single-band unless `bands`
            is given. A composite such as `collection.mean()` has no fixed
            projection and can't be resampled this way, unless its native
            projection is put back first with
            `composite.setDefaultProjection(collection.first().projection())`.
        grid_lons (array-like): Ascending longitudes of the grid cell centres.
        grid_lats (array-like): Ascending latitudes of the grid cell centres.
        cell_size (float): Cell size in degrees (the grid must be regular).
        tile_cells (int): Cells per tile side. Defaults to 200.
        max_source_pixels (int): Most source pixels that may fall in one
            cell. Defaults to 1024, ample for a ~500 m image on a 0.05 degree
            grid (about 144).
        progress (bool): Show a progress bar over the tiles. Defaults to True.
        reducer (ee.Reducer | None): How the source pixels inside a cell are
            combined. Defaults to `ee.Reducer.mean()`. A combined reducer such
            as `ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)`
            gives the cell mean and spread; it renames the bands to
            `<band>_mean` and `<band>_stdDev`, so pass those names in `bands`.
        bands (list[str] | None): Names of the bands of the REDUCED image to
            download, in the order they should come back. `None` (default)
            expects a single-band result and returns a 2-D array. A list
            selects those bands explicitly, so the output order never depends
            on Earth Engine's own band ordering, and returns a 3-D array.
        max_workers (int): Tiles downloaded at the same time, in threads.
            Defaults to 1 (one after the other). The time is almost all
            Earth Engine compute, proportional to the number of cells, so
            parallel requests cut the wall time roughly in proportion;
            keep it well below Earth Engine's concurrent-request limit
            (about 40 for interactive calls).
        retries (int): Extra attempts for a tile whose request fails
            (a transient Earth Engine or network error), waiting 5, 10, ...
            seconds between attempts. Defaults to 0.

    Returns:
        numpy.ndarray: Array of shape `(len(grid_lats), len(grid_lons))` if
            `bands` is None, otherwise `(len(bands), len(grid_lats),
            len(grid_lons))`; float, NaN where the image has no data. Row 0
            is the NORTHERNMOST row (rasters are north-up), so latitude index
            `i` in ascending `grid_lats` is row `len(grid_lats) - 1 - i`.

    Raises:
        ValueError: If the grid axes are empty or not ascending,
            `cell_size`, `tile_cells` or `max_source_pixels` isn't positive,
            `bands` isn't a non-empty list of unique strings, or `max_workers` isn't
            a positive integer or `retries` a non-negative one.
        RuntimeError: If Earth Engine or the download fails for a tile, the
            response doesn't have the expected number of bands, or a tile
            comes back with an unexpected shape.
    """
    lons = np.asarray(grid_lons, dtype=float)
    lats = np.asarray(grid_lats, dtype=float)
    if lons.ndim != 1 or lats.ndim != 1 or lons.size == 0 or lats.size == 0:
        raise ValueError("grid_lons and grid_lats must be non-empty 1-D arrays.")
    if (np.diff(lons) <= 0).any() or (np.diff(lats) <= 0).any():
        raise ValueError("grid_lons and grid_lats must be strictly ascending.")
    if not (cell_size > 0 and tile_cells > 0 and max_source_pixels > 0):
        raise ValueError("cell_size, tile_cells and max_source_pixels must be positive.")
    if bands is not None and (
        not isinstance(bands, (list, tuple))
        or len(bands) == 0
        or not all(isinstance(b, str) for b in bands)
        or len(set(bands)) != len(bands)
    ):
        raise ValueError(f"bands must be a non-empty list of unique band names, got {bands!r}.")

    if not (isinstance(max_workers, int) and max_workers > 0 and isinstance(retries, int) and retries >= 0):
        raise ValueError("max_workers must be a positive integer and retries a non-negative integer.")

    n_bands = 1 if bands is None else len(bands)
    n_rows, n_cols = lats.size, lons.size
    left = lons[0] - cell_size / 2
    top = lats[-1] + cell_size / 2

    try:
        on_grid = image.reduceResolution(
            reducer=ee.Reducer.mean() if reducer is None else reducer,
            maxPixels=max_source_pixels,
        ).reproject(crs="EPSG:4326", crsTransform=[cell_size, 0, left, 0, -cell_size, top])
        if bands is not None:
            on_grid = on_grid.select(list(bands))
    except ee.EEException as exc:
        raise RuntimeError(f"Earth Engine rejected the resampling request: {exc}") from exc

    out = np.full((n_bands, n_rows, n_cols), np.nan)
    tiles = [
        (r0, min(r0 + tile_cells, n_rows), c0, min(c0 + tile_cells, n_cols))
        for r0 in range(0, n_rows, tile_cells)
        for c0 in range(0, n_cols, tile_cells)
    ]

    def fetch_tile(tile_def):
        r0, r1, c0, c1 = tile_def
        x0, y1 = left + c0 * cell_size, top - r0 * cell_size
        transform = [cell_size, 0, x0, 0, -cell_size, y1]
        # The region is pulled in by a quarter cell on every side. Its edges would otherwise sit
        # exactly on pixel edges, where float noise decides whether Earth Engine also returns the
        # neighbouring pixel column/row (a tile one pixel too wide). Every cell centre stays inside.
        inset = cell_size / 4
        bounds = [
            x0 + inset,
            y1 - (r1 - r0) * cell_size + inset,
            x0 + (c1 - c0) * cell_size - inset,
            y1 - inset,
        ]
        for attempt in range(retries + 1):
            try:
                url = on_grid.getDownloadURL(
                    {
                        "name": "tile",
                        "format": "GEO_TIFF",
                        "region": ee.Geometry.Rectangle(bounds, proj="EPSG:4326", geodesic=False),
                        "crs": "EPSG:4326",
                        "crs_transform": transform,
                    }
                )
                response = requests.get(url, timeout=300)
                response.raise_for_status()
                with MemoryFile(response.content) as memfile, memfile.open() as src:
                    if src.count != n_bands:
                        raise RuntimeError(f"expected {n_bands} band(s), got {src.count}")
                    tile = src.read(masked=True).astype(float).filled(np.nan)
                break
            except (ee.EEException, requests.RequestException, rasterio.errors.RasterioError) as exc:
                if attempt < retries:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Failed downloading grid tile rows {r0}-{r1}, cols {c0}-{c1}: {exc}"
                ) from exc
        if tile.shape != (n_bands, r1 - r0, c1 - c0):
            raise RuntimeError(
                f"Tile rows {r0}-{r1}, cols {c0}-{c1} came back with shape {tile.shape}, "
                f"expected {(n_bands, r1 - r0, c1 - c0)}"
            )
        return tile

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # map yields in tile order, so the progress bar and the first error follow the tile list
        for (r0, r1, c0, c1), tile in zip(
            tiles, tqdm(pool.map(fetch_tile, tiles), total=len(tiles), desc="Grid tiles", disable=not progress)
        ):
            out[:, r0:r1, c0:c1] = tile

    return out[0] if bands is None else out
