"""Helpers for the Phase A OpenAQ station census.

Two independent checks, kept deliberately separate:

Recency (is this sensor reporting *right now*) -- a cheap, bulk,
"now"-anchored pre-filter, used only to narrow the candidate list before
the expensive per-sensor pull below:

    a) pull the `LatestResponse` for the PM2.5 parameter (one paginated
       sequence of requests, network-wide -- `get_latest_by_parameter`),
    b) pull the station list for the bounding box of interest (`pull_stations`),
    c) identify each station's PM2.5 sensor (`get_pollutant_sensor_id`),
    d) look up that sensor's most recent reading in the latest-feed table,
    e) keep the station if that reading is within `max_age_days`, else
       discard it (`flag_active_stations`).

This avoids calling `sensors.get()` once per candidate station, which is
what made the original version of this notebook hit a `ServerError` after a
few hundred sequential requests -- everything here is a small, fixed number
of bulk paginated calls, regardless of how many stations are in the bbox.

Period coverage (does this sensor's own reporting period cover a given
date range) -- a separate, "now"-independent question, answered purely
from each sensor's `datetime_first`/`datetime_last` (`pull_sensor_period`,
paced, cached by the caller) via `sensors_covering_period` /
`count_sensors_covering_period`. This is what decides which sensors are
usable for a chosen study period; it has nothing to do with recency, and
deliberately doesn't go through `flag_active_stations` -- an earlier
version bundled both into one `is_active` flag, which was a source of
bugs (a period-coverage question silently also depended on wall-clock
time at call time).
"""

import time
import warnings

import pandas as pd
import numpy as np
from openaq.core.exceptions import ServerError
from openaq.core.responses import LatestResponse


def get_pollutant_id(openaq_client, poll_name="PM25"):
    """Look up the OpenAQ parameter ID for a pollutant by name.

    Parameters:
        openaq_client (openaq.OpenAQ): An authenticated OpenAQ client.
        poll_name (str): Pollutant name to search for, matched
            case-insensitively against OpenAQ parameter names (e.g. "PM25",
            "NO2"). Defaults to "PM25".

    Returns:
        int: The OpenAQ parameter ID corresponding to `poll_name`.

    Raises:
        RuntimeError: If the pollutant list cannot be fetched from OpenAQ.
        LookupError: If no pollutant matching `poll_name` is found.
    """
    try:
        pollutant_list = openaq_client.parameters.list(
            parameter_type="pollutant"
        ).dict()["results"]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch the pollutant list from OpenAQ: {exc}"
        ) from exc

    try:
        pollutant_id = next(
            x["id"] for x in pollutant_list if x["name"].upper() == poll_name.upper()
        )
    except StopIteration:
        raise LookupError(
            f"No OpenAQ pollutant named {poll_name!r} was found "
            f"({len(pollutant_list)} pollutants returned)."
        ) from None

    return pollutant_id


def pull_stations(openaq_client, bbox, pollutant_id=2, limit=1000):
    """Fetch all OpenAQ stations within a bounding box that measure a pollutant.

    Pages through the OpenAQ locations endpoint until a page smaller than
    `limit` is returned, indicating the last page has been reached.

    Parameters:
        openaq_client (openaq.OpenAQ): An authenticated OpenAQ client.
        bbox: Bounding box passed through to `locations.list`, in the
            format expected by the OpenAQ API (min lon, min lat, max lon,
            max lat).
        pollutant_id (int): OpenAQ parameter ID to filter stations by.
            Defaults to 2 (PM2.5).
        limit (int): Maximum number of results to request per page.
            Defaults to 1000.

    Returns:
        tuple[pd.DataFrame, list]: A flattened station table (one row per
            station, dotted column names from the nested API response) and
            the matching list of raw station objects (needed downstream to
            reach each station's `sensors` list).

    Raises:
        RuntimeError: If a page of results cannot be fetched from OpenAQ.
    """
    pages = []
    stations_list = []
    page = 1
    try:
        while True:
            stl = openaq_client.locations.list(
                parameters_id=pollutant_id,
                bbox=bbox,
                page=page,
                limit=limit,
            )
            pages.append(pd.json_normalize(stl.dict()["results"]))
            stations_list.extend(stl.results)
            if len(stl.results) < limit:
                break
            page += 1
            time.sleep(0.1)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch page {page} of the stations list from OpenAQ: {exc}"
        ) from exc
    stations_df = pd.concat(pages, ignore_index=True)
    return stations_df, stations_list


def _get_pollutant_sensor(station, pollutant_id):
    """Return a station's `Sensor` object for a given pollutant, or None.

    Parameters:
        station: A station object as returned in `pull_stations`'
            `stations_list`, exposing a `sensors` list whose items expose
            `parameter.id`.
        pollutant_id (int): OpenAQ parameter ID to match against.

    Returns:
        openaq.core.responses.Sensor | None: The matching sensor object, or
            None if the station has no sensor for `pollutant_id`.
    """
    return next(
        (sensor for sensor in station.sensors if sensor.parameter.id == pollutant_id),
        None,
    )


def get_pollutant_sensor_id(station, pollutant_id):
    """Return a station's sensor ID for a given pollutant, or None if absent.

    Parameters:
        station: A station object as returned in `pull_stations`'
            `stations_list`, exposing a `sensors` list whose items expose
            `id` and `parameter.id`.
        pollutant_id (int): OpenAQ parameter ID to match against.

    Returns:
        int | None: The matching sensor ID, or None if the station has no
            sensor for `pollutant_id`.
    """
    sensor = _get_pollutant_sensor(station, pollutant_id)
    return sensor.id if sensor is not None else None


def get_latest_by_parameter(openaq_client, parameters_id, limit=1000):
    """Bulk-fetch the most recent reading for every sensor of a given parameter.

    Pages through `GET /parameters/{parameters_id}/latest`, the network-wide
    endpoint that returns one row per sensor with its latest measurement --
    the scalable way to check sensor activity across thousands of sensors in
    a handful of requests, instead of calling `sensors.get()` once per
    sensor. This is network-wide, not bounded to any region: the SDK's
    `parameters.latest()` wrapper doesn't support a bbox filter (nor does
    the endpoint itself), so the region filtering happens afterwards, by
    joining this table against a bbox-filtered station list on `sensor_id`
    (see `flag_active_stations`).

    The SDK's `parameters.latest()` wrapper also doesn't expose pagination,
    so this calls the client's underlying `_get` directly and parses the
    response with the same `LatestResponse` model the SDK uses internally.

    Parameters:
        openaq_client (openaq.OpenAQ): An authenticated OpenAQ client.
        parameters_id (int): OpenAQ parameter ID (e.g. the PM2.5 ID from
            `get_pollutant_id`).
        limit (int): Results requested per page. Defaults to 1000.

    Returns:
        pd.DataFrame: One row per sensor, with columns `sensor_id`,
            `location_id`, `latitude`, `longitude`, `value`,
            `datetime_utc`, `datetime_local`.

    Raises:
        RuntimeError: If a page of results cannot be fetched from OpenAQ.
    """
    rows = []
    page = 1
    try:
        while True:
            resp = openaq_client._get(
                f"/parameters/{parameters_id}/latest",
                params={"page": page, "limit": limit},
            )
            latest = LatestResponse.read_response(resp)
            for rec in latest.results:
                rows.append(
                    {
                        "sensor_id": rec.sensors_id,
                        "location_id": rec.locations_id,
                        "latitude": (
                            rec.coordinates.latitude if rec.coordinates else None
                        ),
                        "longitude": (
                            rec.coordinates.longitude if rec.coordinates else None
                        ),
                        "value": rec.value,
                        "datetime_utc": rec.datetime.utc if rec.datetime else None,
                        "datetime_local": rec.datetime.local if rec.datetime else None,
                    }
                )
            if len(latest.results) < limit:
                break
            page += 1
            time.sleep(0.1)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch page {page} of the latest-by-parameter feed from OpenAQ: {exc}"
        ) from exc
    return pd.DataFrame(rows)


def pull_sensor_period(openaq_client, sensor_list, on_sensor_done=None):
    """Fetch each sensor's reporting period (`datetime_first`/`datetime_last`).

    `station.sensors` entries (as returned by `pull_stations`'
    `locations.list` call) are lightweight `SensorBase` objects -- id, name,
    and parameter only, no `datetime_first`/`datetime_last`. Only the full
    `Sensor` object returned by `GET /sensors/{id}` carries that, and
    there's no bulk endpoint for it (the network-wide
    `/parameters/{id}/latest` used by `get_latest_by_parameter` gives only
    the *latest* reading per sensor). So this calls `sensors.get()` once
    per sensor in `sensor_list`, paced at 1 request/s to respect OpenAQ's
    rate limit -- the same pattern `pull_measurements` uses for the same
    reason.

    A per-sensor `openaq.core.exceptions.ServerError` (and its subclasses --
    `BadGatewayError`, `ServiceUnavailableError`, `GatewayTimeoutError`; the
    5xx-class errors OpenAQ occasionally returns for an individual sensor)
    is treated as transient: it's logged via `warnings.warn` and that
    sensor is recorded with `datetime_first`/`datetime_last` both None, and
    the pull continues -- one bad sensor shouldn't abort a pull of hundreds.
    Any other failure still raises immediately rather than silently
    skipping a sensor, since it more likely means something's actually
    wrong (bad ID, auth, rate limit). Either way, no sensor fetched before
    a raised failure is lost: pass `on_sensor_done` to persist each result
    as it comes in, so a caller that caches to disk (e.g. appending to a
    CSV) doesn't have to re-pull sensors already fetched in a prior,
    interrupted run.

    Parameters:
        openaq_client (openaq.OpenAQ): An authenticated OpenAQ client.
        sensor_list (list[int]): OpenAQ sensor IDs to fetch.
        on_sensor_done (callable | None): Optional callback invoked as
            `on_sensor_done(sensor_id, row)` immediately after each sensor
            is successfully fetched, where `row` is a dict with
            `datetime_first`/`datetime_last` keys (same shape as a row of
            the returned DataFrame). Called before the rate-limit sleep, so
            a slow callback doesn't itself violate the pacing.

    Returns:
        tuple[pd.DataFrame, dict]:
            - pd.DataFrame indexed by sensor ID, with columns
              `datetime_first` and `datetime_last` (UTC ISO strings, or
              None where the sensor has no recorded value for either, or
              where the sensor was skipped after a `ServerError`).
            - dict mapping each *successfully* fetched sensor ID to its raw
              `SensorsResponse` object, mirroring `pull_measurements`'
              return shape so a downstream parsing failure doesn't require
              re-fetching from the API. Sensors skipped after a
              `ServerError` have no entry.

    Raises:
        RuntimeError: If a given sensor's details can't be fetched from
            OpenAQ for a reason other than a transient `ServerError`.
        ValueError: If `sensor_list` contains the same sensor ID more than
            once -- fetching it twice would silently discard the first
            response, since both would be stored under the same dict key.
    """
    resp_dict = {}
    rows = []
    for loopn, sensor_id in enumerate(sensor_list):
        if sensor_id in resp_dict:
            raise ValueError(
                f"sensor_list contains duplicate sensor ID {sensor_id!r} "
                f"(at position {loopn}); pulling it again would silently "
                f"overwrite the response already fetched for it."
            )
        if loopn % 50 == 0 and loopn > 0:
            print(f"Pulling sensor {loopn} of {len(sensor_list)}...")
        try:
            resp = openaq_client.sensors.get(sensor_id)
        except ServerError as exc:
            warnings.warn(
                f"Sensor {sensor_id} ({loopn + 1} of {len(sensor_list)}) "
                f"skipped after a server-side error from OpenAQ: {exc}"
            )
            resp = None
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch sensor {sensor_id} ({loopn + 1} of "
                f"{len(sensor_list)}) from OpenAQ: {exc}"
            ) from exc
        else:
            resp_dict[sensor_id] = resp

        sensor = resp.results[0] if resp is not None and resp.results else None
        row = {
            "datetime_first": (
                sensor.datetime_first.utc
                if sensor is not None and sensor.datetime_first is not None
                else None
            ),
            "datetime_last": (
                sensor.datetime_last.utc
                if sensor is not None and sensor.datetime_last is not None
                else None
            ),
        }
        rows.append({"sensor_id": sensor_id, **row})
        if on_sensor_done is not None:
            on_sensor_done(sensor_id, row)
        time.sleep(1)  # 1 request/s

    period_df = pd.DataFrame(rows).set_index("sensor_id")
    return period_df, resp_dict


def sensors_covering_period(sensor_period_df, date_end, n_months=None, date_start=None):
    """Return sensors whose reporting period fully covers [date_start, date_end].

    A sensor is included if `datetime_first` is on/before `date_start` AND
    `datetime_last` is on/after `date_end` -- i.e. it was already reporting
    by the window's start and was still reporting at its end. Sensor-only:
    no station/latest-feed join, no "now" reference (see the module
    docstring for why this is kept separate from `flag_active_stations`).

    Parameters:
        sensor_period_df (pd.DataFrame): Indexed by sensor ID, with
            `datetime_first`/`datetime_last` columns -- the shape returned
            by `pull_sensor_period`.
        date_end (str | pd.Timestamp): End of the coverage window.
        n_months (int | None): Number of months before `date_end` the
            window should start. 0 is allowed (a zero-width window, i.e.
            "is this sensor still reporting exactly at `date_end`").
            Mutually exclusive with `date_start`; exactly one of the two
            must be given.
        date_start (str | pd.Timestamp | None): Explicit start of the
            coverage window. Mutually exclusive with `n_months`.

    Returns:
        pd.DataFrame: The subset of `sensor_period_df` (same columns,
            indexed by sensor ID) satisfying the full-coverage check above.

    Raises:
        ValueError: If both or neither of `n_months`/`date_start` are
            given, if a date can't be parsed, or if the resulting window
            start is after its end.
    """
    if (n_months is None) == (date_start is None):
        raise ValueError(
            "Provide exactly one of n_months or date_start, not both or "
            "neither."
        )

    try:
        date_end_ts = pd.Timestamp(date_end, tz="UTC")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid date_end={date_end!r}: {exc}") from exc

    if date_start is not None:
        try:
            date_start_ts = pd.Timestamp(date_start, tz="UTC")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid date_start={date_start!r}: {exc}") from exc
    else:
        date_start_ts = date_end_ts - pd.DateOffset(months=n_months)

    if date_start_ts > date_end_ts:
        raise ValueError(
            f"date_start ({date_start_ts}) must be on or before date_end "
            f"({date_end_ts})."
        )

    first = pd.to_datetime(sensor_period_df["datetime_first"], utc=True, errors="coerce")
    last = pd.to_datetime(sensor_period_df["datetime_last"], utc=True, errors="coerce")

    covers = first.notna() & last.notna() & first.le(date_start_ts) & last.ge(date_end_ts)
    return sensor_period_df.loc[covers]


def count_sensors_covering_period(sensor_period_df, date_end, n_months=None, date_start=None):
    """Count sensors whose reporting period fully covers [date_start, date_end].

    Thin wrapper around `sensors_covering_period` -- see it for the
    full-coverage definition and parameters. Use this when only the count
    is needed; use `sensors_covering_period` directly when the matching
    sensor IDs themselves are needed (e.g. to slice a station table).

    Returns:
        int: Number of sensors satisfying the full-coverage check.
    """
    return len(
        sensors_covering_period(
            sensor_period_df, date_end, n_months=n_months, date_start=date_start
        )
    )


def flag_active_stations(
    stations_df,
    stations_list,
    pollutant_id,
    latest_df,
    max_age_days=365,
    as_of=None,
):
    """Flag which stations' pollutant sensor has reported recently.

    Joins each station's pollutant-specific sensor ID against the bulk
    latest-by-parameter feed (`get_latest_by_parameter`) to check how long
    ago it last reported, without a per-sensor API round trip. Uses the
    feed's UTC timestamp for the age comparison (not the local one) so the
    result doesn't depend on each station's own timezone offset.

    This is a recency check only, anchored to `as_of` ("now" by default) --
    it says nothing about whether a sensor's reporting period covers any
    particular study window. For that, see `sensors_covering_period` /
    `count_sensors_covering_period`, which work off `pull_sensor_period`'s
    output and have no "now" dependency at all. Keeping the two separate
    is deliberate: an earlier version of this function folded a
    `date_from` coverage check into `is_active` too, which meant a
    period-coverage result silently also depended on wall-clock time at
    call time -- confusing, and a source of bugs.

    Parameters:
        stations_df (pd.DataFrame): Station table as returned by
            `pull_stations`.
        stations_list (list): Station objects aligned row-for-row with
            `stations_df`, as returned by `pull_stations`.
        pollutant_id (int): OpenAQ parameter ID identifying which sensor on
            each station to check.
        latest_df (pd.DataFrame): Output of `get_latest_by_parameter` for
            the same `pollutant_id`.
        max_age_days (int): Maximum age, in days, of a sensor's most recent
            reading for the station to be flagged active. Defaults to 365.
        as_of (pd.Timestamp | None): Timezone-aware reference time to
            measure recency against. Defaults to `pd.Timestamp.now(tz="UTC")`.

    Returns:
        pd.DataFrame: A copy of `stations_df` with added columns
            `pollutant_sensor_id`, `datetime_last_utc`, `days_since_last`,
            and `is_active` (recency only).

    Raises:
        ValueError: If `stations_df` and `stations_list` aren't
            row-aligned.
    """
    if len(stations_df) != len(stations_list):
        raise ValueError(
            f"stations_df ({len(stations_df)} rows) and stations_list "
            f"({len(stations_list)} items) must be row-aligned."
        )

    as_of = as_of or pd.Timestamp.now(tz="UTC")

    out = stations_df.copy()
    out["pollutant_sensor_id"] = [
        get_pollutant_sensor_id(station, pollutant_id) for station in stations_list
    ]

    latest_lookup = latest_df.drop_duplicates("sensor_id").set_index("sensor_id")[
        "datetime_utc"
    ]
    out["datetime_last_utc"] = out["pollutant_sensor_id"].map(latest_lookup)

    parsed_last = pd.to_datetime(out["datetime_last_utc"], utc=True, errors="coerce")
    out["days_since_last"] = (as_of - parsed_last).dt.days
    out["is_active"] = out["days_since_last"].notna() & out["days_since_last"].le(
        max_age_days
    )
    return out


def pull_measurements(
    openaq_client,
    sensor_list,
    data="days",
    rollup="monthly",
    date_from="2024-01-01",
    date_to="2025-12-31",
    limit=25,
):
    """Fetch raw measurement time series for a list of sensors from OpenAQ.

    Calls `measurements.list` once per sensor ID, sleeping briefly between
    calls to respect OpenAQ's rate limit. Returns the raw SDK response
    objects (one per sensor) rather than already-converted DataFrames, so
    that a downstream conversion failure doesn't require re-fetching from
    the API -- see `make_df_from_measurement` / `concat_measurements`.

    Parameters:
        openaq_client (openaq.OpenAQ): An authenticated OpenAQ client.
        sensor_list (list[int]): OpenAQ sensor IDs to fetch measurements for.
        data (str): Temporal resolution passed to `measurements.list`
            (e.g. "days", "hours", "years"). Defaults to "days".
        rollup (str): Temporal rollup passed to `measurements.list` (e.g.
            "monthly"). Defaults to "monthly".
        date_from (str): Start date (ISO format) of the query window.
            Defaults to "2024-01-01".
        date_to (str): End date (ISO format) of the query window. Defaults
            to "2025-12-31".
        limit (int): Maximum number of periods requested per sensor.
            Defaults to 25.

    Returns:
        dict: Maps each sensor ID in `sensor_list` to its measurement
            response object, as returned by `measurements.list`. Downstream,
            `concat_measurements` uses these sensor IDs as the outer level
            of a `(sensor_id, time_local)` MultiIndex. Local is used for
            downstream compatibility with TROPOMI measurements.

    Raises:
        RuntimeError: If a given sensor's measurements cannot be fetched
            from OpenAQ.
        ValueError: If `sensor_list` contains the same sensor ID more than
            once -- fetching it twice would silently discard the first
            result when both are stored under the same dict key.
    """
    meas_dict = {}
    for loopn, sensor_id in enumerate(sensor_list):
        if sensor_id in meas_dict:
            raise ValueError(
                f"sensor_list contains duplicate sensor ID {sensor_id!r} "
                f"(at position {loopn}); pulling it again would silently "
                f"overwrite the measurements already fetched for it."
            )
        if loopn % 50 == 0 and loopn > 0:
            print(f"Pulling sensor {loopn} of {len(sensor_list)}...")
        try:
            measurement = openaq_client.measurements.list(
                sensors_id=sensor_id,
                data=data,
                rollup=rollup,
                date_from=date_from,
                date_to=date_to,
                limit=limit,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch measurements for sensor {sensor_id} "
                f"({loopn + 1} of {len(sensor_list)}) from OpenAQ: {exc}"
            ) from exc
        meas_dict[sensor_id] = measurement
        time.sleep(1)  # 1 request/s
    return meas_dict


def normalize_unit(unit: str) -> str:
    """Normalize a measurement-unit string into a compact ASCII token.

    Examples:
        'µg/m³'  -> 'ugm3'
        'μg/m3'  -> 'ugm3'
        'ng/m³'  -> 'ngm3'
        'mg/m³'  -> 'mgm3'
        'ppm'    -> 'ppm'
    """
    import unicodedata
    import re
    # Decompose unicode -- turns superscripts like ³ into ASCII '3', ² into '2', etc.
    s = unicodedata.normalize('NFKD', unit)

    # Explicit replacements for characters NFKD doesn't reduce on its own
    replacements = {
        'µ': 'u',   # micro sign (U+00B5)
        'μ': 'u',   # greek small letter mu (U+03BC) -- two visually identical
                    # but distinct unicode characters; both show up in the wild
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    # Strip everything that isn't a letter or digit (slashes, spaces, carets, etc.)
    s = re.sub(r'[^a-zA-Z0-9]', '', s)

    return s.lower()


def make_df_from_measurement(measurement, idx=0):
    """Convert one OpenAQ measurement response into a tidy, time-indexed DataFrame.

    Flattens the nested `results` list of a `measurements.list` response
    (as returned by `pull_measurements`) with `pd.json_normalize`, then
    selects and renames the columns needed downstream and sets a UTC
    datetime index from each period's start time.

    Some sensors have no data for the requested window, in which case
    `results` is an empty list; `pd.json_normalize` on an empty list
    produces a DataFrame with *no columns at all*, so the expected
    columns are checked for explicitly and an empty result is handled
    before any column lookup is attempted.

    Parameters:
        measurement: A measurement response object as returned by
            `pull_measurements`, exposing a `.dict()` method with a
            `results` list of period records.
        idx: Identifier for the sensor/station this measurement belongs
            to -- typically the sensor ID key from the `meas_dict` returned
            by `pull_measurements`. Stored on the returned DataFrame's
            `.name` attribute for traceability if this function is used on
            its own; `concat_measurements` doesn't rely on it, since it
            labels each measurement's rows via `pd.concat(..., keys=...)`
            instead. Defaults to 0.

    Returns:
        pd.DataFrame: Indexed by local (not UTC) period-start datetime as a
            naive `datetime64` (see note above on why the offset is
            stripped), with columns `concentration`, `stdev`, `unit`,
            `percent_coverage`, and `utc_offset` (the offset that was
            stripped from the index, as e.g. `"+0530"`, kept in case UTC
            needs to be recovered later). Empty (no rows) if `measurement`
            has no results for the requested window.

    Raises:
        RuntimeError: If `measurement` doesn't expose a `.dict()` method
            with the expected `results` key, or if the period start time
            can't be parsed as a datetime.
        KeyError: If `results` is non-empty but is missing one of the
            columns this function expects after flattening (e.g. the
            OpenAQ response shape changed).
    """
    try:
        results = measurement.dict()["results"]
    except Exception as exc:
        raise RuntimeError(
            f"Measurement for idx={idx!r} could not be read: expected an "
            f"object with a .dict()['results'] list, got {type(measurement)!r} "
            f"({exc})"
        ) from exc

    columns = {
        "summary.avg": "concentration",
        "summary.sd" : "stdev",
        "parameter.units": "unit",
        "period.datetime_from.local": "time_local",
        "coverage.percent_complete": "percent_coverage",
    }

    if not results:
        empty = pd.DataFrame(columns=list(columns.values()) + ["utc_offset"])
        empty = empty.set_index(pd.DatetimeIndex([], name="time_local"))
        empty = empty[
            ["concentration", "stdev", "unit", "percent_coverage", "utc_offset"]
        ]
        empty.name = idx
        return empty

    meas = pd.json_normalize(results)

    missing = [col for col in columns if col not in meas.columns]
    if missing:
        raise KeyError(
            f"Measurement for idx={idx!r} is missing expected column(s) "
            f"{missing} after json_normalize (available columns: "
            f"{list(meas.columns)}); the OpenAQ response shape may have "
            f"changed."
        )

    meas = meas.loc[:, list(columns)].rename(columns=columns)

    try:
        local_dt = pd.to_datetime(meas["time_local"], utc=False)
        # Different sensors report in different local UTC offsets (e.g. a
        # station in India vs. one in Pakistan); pandas has no single
        # datetime64[tz] dtype that can hold more than one fixed offset at
        # once, so concatenating tz-aware columns like that across sensors
        # (see concat_measurements) silently downgrades them to `object`
        # dtype, breaking `.dt` downstream. Stripping the offset here keeps
        # the correct local wall-clock reading as a naive datetime64, which
        # survives concatenation across sensors/timezones intact. The
        # offset itself is saved separately (as e.g. "+0530") in case it's
        # ever needed to recover UTC -- captured per-row rather than once
        # per sensor, since it can differ within a single sensor's own
        # rows across a DST transition.
        if local_dt.dt.tz is not None:
            meas["utc_offset"] = local_dt.dt.strftime("%z")
            local_dt = local_dt.dt.tz_localize(None)
        else:
            meas["utc_offset"] = None
        meas = meas.set_index(local_dt)
    except Exception as exc:
        raise RuntimeError(
            f"Measurement for idx={idx!r} has an unparseable 'time_local' "
            f"column: {exc}"
        ) from exc

    meas = meas[["concentration", "stdev", "unit", "percent_coverage", "utc_offset"]]
    # Normalize units
    try:
        meas["unit"] = meas["unit"].apply(normalize_unit)
    except Exception as exc:
        pass
    meas.name = idx
    return meas


def concat_measurements(meas_dict):
    """Convert a dict of OpenAQ measurement responses into one combined DataFrame.

    Converts each measurement in `meas_dict` with `make_df_from_measurement`,
    then concatenates the results into a single DataFrame with a
    `(sensor_id, time_local)` MultiIndex, so rows stay traceable to their
    source sensor and downstream code can `.groupby(level="sensor_id")` or
    reindex per sensor.

    Parameters:
        meas_dict (dict): Maps sensor ID to measurement response object,
            as returned by `pull_measurements`.

    Returns:
        pd.DataFrame: All measurements concatenated, with columns
            `concentration`, `stdev`, `unit`, `percent_coverage`,
            `utc_offset` and a `(sensor_id, time_local)` MultiIndex. Empty
            (but still carrying that MultiIndex shape) if `meas_dict` is
            empty or every measurement in it had no results.

    Raises:
        TypeError: If `meas_dict` isn't a dict (e.g. a plain list of
            measurements, as `pull_measurements` used to return).
        RuntimeError: If any measurement in `meas_dict` cannot be
            converted (see `make_df_from_measurement`).
        KeyError: If any measurement is missing an expected column (see
            `make_df_from_measurement`).
    """
    if not isinstance(meas_dict, dict):
        raise TypeError(
            f"concat_measurements expects a dict mapping sensor ID to "
            f"measurement (as returned by pull_measurements), got "
            f"{type(meas_dict)!r}."
        )

    if not meas_dict:
        empty_index = pd.MultiIndex.from_arrays(
            [[], []], names=["pm25_sensor_id", "time_local"]
        )
        return pd.DataFrame(
            columns=["concentration", "stdev", "unit", "percent_coverage", "utc_offset"],
            index=empty_index,
        )

    df_list = [
        make_df_from_measurement(meas, idx=s_id) for s_id, meas in meas_dict.items()
    ]
    return pd.concat(df_list, keys=meas_dict.keys(), names=["pm25_sensor_id"])
