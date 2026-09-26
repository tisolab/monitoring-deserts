"""Helpers for the Phase E ERA5 meteorology pull via the Copernicus Climate Data Store (CDS).

`build_request` builds a normalized request dict for the monthly-averaged ERA5
product; `pull_era5` sends it through an already-initialized `cdsapi.Client` and
writes the result to disk. See the project plan, Section 5.3, for `~/.cdsapirc`
setup, the CDS/ADS mixup trap, and the ZIP/`expver` quirks that get handled
downstream once the file is on disk (not in this module).

`bbox` here follows CDS's own `[North, West, South, East]` convention for the
`area` request field -- deliberately different from this project's usual
`(min_lon, min_lat, max_lon, max_lat)` order used elsewhere (e.g.
`census_scope.yaml`'s `bounding_box`). Reorder explicitly at the call site;
don't pass one of those tuples straight through.
"""


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
