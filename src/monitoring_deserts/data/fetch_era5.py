"""Helpers for the Phase E ERA5 meteorology pull via the Copernicus Climate Data Store (CDS).

`build_request` builds a normalized request dict for the monthly-averaged ERA5
product; `pull_era5` sends it through an already-initialized `cdsapi.Client` and
writes the result to disk. See the project plan, Section 5.3, for `~/.cdsapirc`
setup, the CDS/ADS mixup trap, and the ZIP/`expver` quirks that get handled
downstream once the file is on disk. `open_dataset` is that downstream step --
it opens whatever `pull_era5` wrote, transparently handling the CDS quirks
described there (a ZIP archive under the requested filename, instantaneous
and accumulated variables split across separate files with mismatched
hour-of-day metadata, and a stray `expver` dimension on requests that
straddle ERA5's final/preliminary release boundary).

`bbox` here follows CDS's own `[North, West, South, East]` convention for the
`area` request field -- deliberately different from this project's usual
`(min_lon, min_lat, max_lon, max_lat)` order used elsewhere (e.g.
`census_scope.yaml`'s `bounding_box`). Reorder explicitly at the call site;
don't pass one of those tuples straight through.

CDS/MARS requests take `year` and `month` as two independent lists and
always request their full cross product within a single call -- there's no
way to pass paired (year, month) values directly. A study period that
doesn't start in January and/or end in December (e.g. 2025-03 through
2026-07) therefore can't be expressed as one request without also pulling
months outside that range -- including, worst case, months CDS hasn't
published yet, which fails the *entire* job rather than just those months
(see `pull_era5`'s docstring). `build_year_month_groups` splits such a
range into one (year, months) pair per calendar year it spans, so each
resulting request's cross product is exactly the months wanted.
"""

import zipfile
from pathlib import Path

import pandas as pd
import xarray as xr


def build_year_month_groups(day_start, day_end):
    """Group a date range into one (year, months) pair per calendar year it spans.

    CDS/MARS requests take `year` and `month` as two independent lists and
    always request their full cross product in a single call -- there's no
    way to pass paired (year, month) values directly. Splitting into one
    request per calendar year, each with only that year's own months, is the
    standard way around this: within a single year, the cross product of
    `[year]` and that year's months is exactly the months wanted, with no
    overshoot into months outside `day_start`/`day_end` (and no risk of
    requesting months CDS hasn't published yet, which fails the entire job
    rather than just those months -- see `pull_era5`'s docstring).

    `day_end`'s own calendar month is excluded, matching `build_month_windows`'s
    (`fetch_earthengine.py`) exclusive-of-`DAY_END` convention -- the two
    functions are meant to be called with the same `DAY_START`/`DAY_END`
    pair and produce data covering the same set of months.

    Parameters:
        day_start (str | pd.Timestamp | datetime.datetime): Start of the date
            range. Its calendar month is included in full.
        day_end (str | pd.Timestamp | datetime.datetime): End of the date
            range. Its calendar month is excluded.

    Returns:
        list[tuple[int, list[int]]]: One (year, months) pair per calendar
            year the range spans, sorted by year, each `months` list sorted
            ascending. Pass each pair's `year` (wrapped in a list, e.g.
            `[year]`) and `months` to `build_request` as `years`/`months`,
            and issue one `pull_era5` call per pair.

    Raises:
        ValueError: If `day_start`/`day_end` can't be parsed as dates, or if
            `day_start` is after `day_end`.
    """
    try:
        start_ts = pd.Timestamp(day_start)
        end_ts = pd.Timestamp(day_end)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not parse day_start={day_start!r} / day_end={day_end!r} "
            f"as dates: {exc}"
        ) from exc

    if start_ts > end_ts:
        raise ValueError(f"day_start ({start_ts}) must not be after day_end ({end_ts}).")

    periods = pd.period_range(start_ts, end_ts, freq="M")[:-1]

    groups = {}
    for period in periods:
        groups.setdefault(period.year, []).append(period.month)

    return [(year, sorted(groups[year])) for year in sorted(groups)]


def build_request(
    product_type="monthly_averaged_reanalysis",
    variables=None,
    years=None,
    months=None,
    time="00:00",
    bbox=(29.5, 75.8, 27.1, 78.7),
    data_format="netcdf",
):
    """Build a CDS API request dictionary for a monthly-averaged ERA5 retrieval.

    Parameters:
        product_type (str): CDS product type. Defaults to
            "monthly_averaged_reanalysis" -- the monthly-means product this
            project uses (Section 5.3), dramatically smaller than hourly data
            and matching the project's monthly time resolution.
        variables (list[str] | None): ERA5 variable names to request, e.g.
            "2m_temperature", "boundary_layer_height". Defaults to
            `["boundary_layer_height"]` if not given.
        years (list[int] | None): Years to request. Defaults to `[2024]` if
            not given.
        months (list[int] | None): Months to request, each in 1-12. Defaults
            to `[1]` if not given.
        time (str): Hour of day to request, "HH:MM". Defaults to "00:00" --
            meaningless at monthly resolution but required by the API.
        bbox (list | tuple): Bounding box as `[North, West, South, East]`,
            CDS's own `area` convention.
        data_format (str): Requested output format. Defaults to "netcdf" --
            note CDS has been observed to return a ZIP archive under this
            filename regardless (Section 5.3); handle that after the file is
            downloaded, not here.

    Returns:
        dict: A request dictionary ready to pass to `pull_era5` as `request`.

    Raises:
        ValueError: If `variables`, `years`, or `months` is empty, if any
            month is outside 1-12, or if `bbox` doesn't have exactly 4
            elements or its North/South or East/West bounds are inverted.
    """
    if variables is None:
        variables = ["boundary_layer_height"]
    if years is None:
        years = [2024]
    if months is None:
        months = [1]

    if not variables:
        raise ValueError("variables is empty -- nothing to request.")
    if not years:
        raise ValueError("years is empty -- nothing to request.")
    if not months:
        raise ValueError("months is empty -- nothing to request.")

    bad_months = [m for m in months if not (1 <= m <= 12)]
    if bad_months:
        raise ValueError(f"months must be in 1-12, got: {bad_months}")

    if len(bbox) != 4:
        raise ValueError(
            f"bbox must have exactly 4 elements [North, West, South, East], got {bbox!r}"
        )
    north, west, south, east = bbox
    if north <= south:
        raise ValueError(
            f"bbox's North ({north}) must be greater than South ({south}) -- got bbox={bbox!r}"
        )
    if east <= west:
        raise ValueError(
            f"bbox's East ({east}) must be greater than West ({west}) -- got bbox={bbox!r}"
        )

    req = {
        "product_type": product_type,
        "variable": variables,
        "year": years,
        "month": months,
        "time": time,
        "area": bbox,
        "data_format": data_format,
    }
    return req


def pull_era5(era5_client, target_path, name, request):
    """Retrieve one ERA5 request from the CDS API and save it to disk.

    Parameters:
        era5_client (cdsapi.Client): An initialized CDS API client -- reads
            its credentials from `~/.cdsapirc` (see Section 5.3 for setup).
        target_path (str | Path): Output *file* path (e.g.
            "data/raw/meteorology/era5_monthly.nc") -- passed straight
            through as `cdsapi.Client.retrieve`'s `target` argument.
        name (str): CDS dataset id, e.g.
            "reanalysis-era5-single-levels-monthly-means".
        request (dict): Request dictionary, e.g. from `build_request`.

    Returns:
        None. The retrieved file is written to `target_path` as a side
        effect -- `cdsapi.Client.retrieve` doesn't return the data itself.

    Raises:
        ValueError: If `name` is empty or `request` is empty.
        RuntimeError: If the CDS API rejects the request or the retrieval
            otherwise fails. Unlike `earthengine-api`'s `ee.EEException`,
            `cdsapi` doesn't expose a documented, stable exception type for
            this, so this catches broadly rather than silently missing real
            failures.
    """
    if not name:
        raise ValueError("name is empty -- no CDS dataset id given.")
    if not request:
        raise ValueError("request is empty -- nothing to retrieve.")

    try:
        era5_client.retrieve(name, request, target_path)
    except Exception as exc:
        raise RuntimeError(
            f"CDS API rejected the request for dataset {name!r}: {exc}"
        ) from exc


def open_dataset(filepath):
    """Open an ERA5 file retrieved via `pull_era5`, handling CDS's quirks (Section 5.3).

    CDS sometimes returns a ZIP archive under the requested `.nc` filename,
    inside which instantaneous and accumulated variables can be split across
    separate files with mismatched hour-of-day metadata (e.g. 00:00 vs
    06:00) even though both represent the same monthly mean. When that
    happens, each file's time coordinate is floored to the day before
    merging, so they land on the same, join-friendly timestamp. Separately,
    a request spanning ERA5's final/preliminary release boundary comes back
    with an extra `expver` dimension; that's collapsed with `.combine_first`
    (not `.reduce(np.nansum, ...)`, which would silently turn a genuine data
    gap into 0.0) regardless of whether the file was a ZIP.

    Parameters:
        filepath (str | Path): Path to the file `pull_era5` wrote -- may be
            a plain NetCDF file or a ZIP archive (with either a `.nc` or a
            `.zip` extension; CDS doesn't reliably rename it either way).

    Returns:
        xarray.Dataset: The opened (and, if needed, merged/normalized)
            dataset.

    Raises:
        FileNotFoundError: If `filepath` doesn't exist.
        RuntimeError: If `filepath` is a ZIP archive with no `.nc` files
            inside it, or if opening/merging the file otherwise fails.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} does not exist.")

    try:
        if zipfile.is_zipfile(filepath):
            extract_dir = filepath.with_suffix("")  # drop .nc, use as a folder name
            with zipfile.ZipFile(filepath) as zf:
                zf.extractall(extract_dir)
            nc_files = sorted(extract_dir.glob("*.nc"))
            if not nc_files:
                raise RuntimeError(
                    f"{filepath} is a ZIP archive but contains no .nc files."
                )

            datasets = [xr.open_dataset(f) for f in nc_files]
            time_coord = "valid_time" if "valid_time" in datasets[0].coords else "time"
            normalized = [
                ds.assign_coords({time_coord: ds[time_coord].dt.floor("D")})
                for ds in datasets
            ]
            era5 = xr.merge(normalized, compat="no_conflicts")
        else:
            era5 = xr.open_dataset(filepath, engine="netcdf4")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Failed to open ERA5 file at {filepath}: {exc}") from exc

    if "expver" in era5.dims:
        era5 = era5.sel(expver=1).combine_first(era5.sel(expver=5))

    return era5


def concat_datasets(*datasets):
    """Concatenate multiple ERA5 datasets into one, in chronological order.

    Meant for stitching back together datasets pulled as separate per-year
    requests (see `build_year_month_groups`) -- each one opened separately
    via `open_dataset`, then combined here into a single time series.

    Parameters:
        *datasets (xarray.Dataset): Two or more ERA5 datasets to
            concatenate, e.g. `open_dataset`'s return value for each
            per-year file. Order doesn't matter -- the result is sorted by
            time regardless.

    Returns:
        xarray.Dataset: All input datasets concatenated along their time
            coordinate ("valid_time" or "time", whichever they use) and
            sorted chronologically.

    Raises:
        ValueError: If fewer than 2 datasets are given, if they don't all
            use the same time coordinate name, or if the concatenated
            result has duplicate timestamps (the input datasets overlap in
            time rather than being consecutive, e.g. overlapping year
            groups).
    """
    if len(datasets) < 2:
        raise ValueError(f"Need at least 2 datasets to concatenate, got {len(datasets)}.")

    time_coords = {"valid_time" if "valid_time" in ds.coords else "time" for ds in datasets}
    if len(time_coords) > 1:
        raise ValueError(
            f"Datasets use inconsistent time coordinate names: {sorted(time_coords)} "
            "-- can't concatenate them along a shared time axis."
        )
    time_coord = time_coords.pop()

    combined = xr.concat(datasets, dim=time_coord).sortby(time_coord)

    duplicated = pd.Index(combined[time_coord].values).duplicated()
    if duplicated.any():
        dup_times = combined[time_coord].values[duplicated]
        raise ValueError(
            f"Concatenated dataset has {duplicated.sum()} duplicate timestamp(s): "
            f"{sorted(set(dup_times))} -- input datasets overlap in time rather "
            "than being consecutive."
        )

    return combined


def interpolate(dataset, dataframe, dimname, method="linear"):
    """Interpolate an xarray Dataset onto pointwise (longitude, latitude) coordinates.

    Uses xarray's pointwise indexing (Section 5.3): passing longitude and
    latitude as two `xr.DataArray`s sharing the same new dimension name
    (`dimname`) pairs them up position-by-position -- one point per row of
    `dataframe` -- rather than interpolating onto their cross product.

    Parameters:
        dataset (xarray.Dataset): Must have `longitude`/`latitude`
            coordinates, e.g. an ERA5 dataset from `open_dataset`.
        dataframe (pd.DataFrame): Must have `longitude`/`latitude` columns,
            in plain WGS84 degrees -- one row per point to interpolate onto
            (e.g. `analysis_df`'s stations or `analysis_grid`'s cell
            centroids).
        dimname (str): Name for the new dimension pairing `dataframe`'s rows
            to `dataset`'s interpolated points, e.g. "station" or "cell".
        method (str): Interpolation method passed to
            `xarray.Dataset.interp`. Defaults to "linear".

    Returns:
        xarray.Dataset: `dataset` interpolated onto each row of `dataframe`,
            indexed along `dimname` instead of the original longitude/
            latitude grid.

    Raises:
        ValueError: If `dataframe` is missing `longitude`/`latitude`
            columns, has no rows, has a null or out-of-range longitude/
            latitude, or if any point falls outside `dataset`'s own
            coordinate range -- interpolating there would otherwise
            silently return NaN instead of raising.
        RuntimeError: If `xarray.Dataset.interp` otherwise fails, e.g.
            because `dataset` doesn't have `longitude`/`latitude`
            coordinates at all.
    """
    required_cols = {"longitude", "latitude"}
    missing = required_cols - set(dataframe.columns)
    if missing:
        raise ValueError(f"dataframe is missing required column(s): {sorted(missing)}")
    if dataframe.empty:
        raise ValueError("dataframe has no rows -- nothing to interpolate onto.")

    bad_coords = (
        dataframe["longitude"].isna()
        | dataframe["latitude"].isna()
        | ~dataframe["longitude"].between(-180, 180)
        | ~dataframe["latitude"].between(-90, 90)
    )
    if bad_coords.any():
        raise ValueError(
            f"{int(bad_coords.sum())} row(s) have a missing or out-of-range "
            "longitude/latitude."
        )

    lon_min, lon_max = float(dataset.longitude.min()), float(dataset.longitude.max())
    lat_min, lat_max = float(dataset.latitude.min()), float(dataset.latitude.max())
    out_of_bounds = ~dataframe["longitude"].between(lon_min, lon_max) | ~dataframe[
        "latitude"
    ].between(lat_min, lat_max)
    if out_of_bounds.any():
        raise ValueError(
            f"{int(out_of_bounds.sum())} row(s) fall outside dataset's coordinate "
            f"range (longitude [{lon_min}, {lon_max}], latitude [{lat_min}, "
            f"{lat_max}]) -- interpolating there would silently return NaN."
        )

    try:
        return dataset.interp(
            longitude=xr.DataArray(dataframe.longitude, dims=dimname),
            latitude=xr.DataArray(dataframe.latitude, dims=dimname),
            method=method,
        )
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Failed to interpolate dataset onto dataframe: {exc}") from exc


def cleanup_dataset(dataset):
    """Rename ERA5's raw variable codes and add a `month` coordinate for downstream merges.

    ERA5 datasets come back with cryptic short variable names (`t2m`, `tp`,
    ...) and a `valid_time`/`time` coordinate. This renames the variables to
    the project's own descriptive names, and swaps in a `month` dimension
    (a plain `datetime64`, one value per original timestep) in place of
    `valid_time`/`time`.

    `month` is deliberately kept as `datetime64` here, not converted to a
    `pandas.Period` (the format `gtruth_df`/`satellite_df`/`analysis_df` use
    for their own `month` columns, via `.dt.to_period("M")` in notebooks
    02/03) -- `Period` is a pandas-only extension dtype that NetCDF's CF
    encoder can't serialize at all (`to_netcdf` raises `TypeError: Cannot
    interpret 'period[M]' as a data type`). Do the `.dt.to_period("M")`
    conversion downstream, after `.to_dataframe()`, the same way notebook 03
    does for `satellite_df` -- once this leaves `xarray` for good.

    Parameters:
        dataset (xarray.Dataset): An ERA5 dataset, e.g. from `open_dataset`
            or `concat_datasets`, with the raw ERA5 short variable names
            still in place (`tp`, `t2m`, `d2m`, `u10`, `v10`, `blh`, `sp`)
            and a `valid_time` or `time` coordinate.

    Returns:
        xarray.Dataset: `dataset` with variables renamed to
            `total_precipitation`, `temperature`, `dew_point`, `u_wind`,
            `v_wind`, `boundary_layer_height`, `surface_pressure`, indexed
            by a new `month` dimension (`datetime64`) in place of
            `valid_time`/`time`.

    Raises:
        ValueError: If `dataset` is missing any of the expected raw
            variable names, or has neither a `valid_time` nor a `time`
            coordinate.
    """
    new_names = {
        "tp": "total_precipitation",
        "t2m": "temperature",
        "d2m": "dew_point",
        "u10": "u_wind",
        "v10": "v_wind",
        "blh": "boundary_layer_height",
        "sp": "surface_pressure",
    }

    missing = set(new_names) - set(dataset.data_vars)
    if missing:
        raise ValueError(
            f"dataset is missing expected variable(s): {sorted(missing)} -- "
            f"has {sorted(dataset.data_vars)}."
        )

    if "valid_time" in dataset.coords:
        time_coord = "valid_time"
    elif "time" in dataset.coords:
        time_coord = "time"
    else:
        raise ValueError("dataset has neither a 'valid_time' nor a 'time' coordinate.")

    ds = dataset.rename(new_names)
    month_values = pd.to_datetime(ds[time_coord].values)
    ds = ds.assign_coords(month=(time_coord, month_values))
    ds = ds.swap_dims({time_coord: "month"})
    return ds




