import os
import dotenv
import time
from pathlib import Path
from datetime import datetime

from openaq import OpenAQ
from numpy import argmax
import pandas as pd
import geopandas


def check_measurement_in_period(sensor, start="2024-01-01", end="2026-01-01"):
    """Check whether an OpenAQ sensor's reporting coverage spans a date range.

    Parameters:
        sensor: The result of an ``openaq.OpenAQ.Client.sensors.get()``
            query, expected to expose ``datetime_first`` and
            ``datetime_last`` attributes with a ``.local`` ISO timestamp.
        start (str): ISO date/datetime string marking the start of the
            required coverage window. Defaults to "2024-01-01".
        end (str): ISO date/datetime string marking the end of the
            required coverage window. Defaults to "2026-01-01".

    Returns:
        bool: True if the sensor's first and last measurement dates fully
            cover [start, end]; False if the sensor has no coverage
            information or its dates cannot be parsed.

    Raises:
        ValueError: If `start` or `end` cannot be parsed as ISO dates.
    """
    try:
        date_start = datetime.fromisoformat(start).date()
        date_end = datetime.fromisoformat(end).date()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid start/end date passed to check_measurement_in_period: "
            f"start={start!r}, end={end!r}"
        ) from exc

    # Guard the coverage attributes before dereferencing `.local`; a sensor
    # may report either bound as None when it has never produced data.
    if getattr(sensor, "datetime_first", None) is None:
        return False
    if getattr(sensor, "datetime_last", None) is None:
        return False

    try:
        sensor_first = datetime.fromisoformat(sensor.datetime_first.local).date()
        sensor_last = datetime.fromisoformat(sensor.datetime_last.local).date()
    except (AttributeError, TypeError, ValueError):
        return False

    if sensor_first <= date_start and sensor_last >= date_end:
        return True
    else:
        return False


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
        pd.DataFrame: MODIFY.
        list: MODIFY

    Raises:
        RuntimeError: If a page of results cannot be fetched from OpenAQ.
    """
    pages = []
    stations_list = []
    page = 1
    lim = limit
    try:
        while True:  # Loop until reaching the final page
            stl = openaq_client.locations.list(
                parameters_id=pollutant_id,
                bbox=bbox,
                page=page,
                limit=lim,
            )
            pages.append(pd.json_normalize(stl.dict()["results"]))
            stations_list.extend(stl.results)
            if len(stl.results) < lim:
                break
            page += 1
            time.sleep(0.1)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch page {page} of the stations list from OpenAQ: {exc}"
        ) from exc
    stations_df = pd.concat(pages, ignore_index=True)
    return stations_df, stations_list


def get_active_sensors(openaq_client, stations_list, pollutant_id, start, end):
    """Find sensors that measured a pollutant throughout a date range.

    For each station, selects the sensor that measures `pollutant_id` and
    keeps it if its reported coverage spans [start, end], as determined by
    `check_measurement_in_period`.

    Parameters:
        openaq_client (openaq.OpenAQ): An authenticated OpenAQ client.
        stations_list (list): Station objects (e.g. as returned by
            `pull_stations`), each exposing a `sensors` list whose items
            have `id` and `parameters.id` attributes.
        pollutant_id (int): OpenAQ parameter ID identifying which sensor
            on each station to check.
        start (str): ISO date/datetime string marking the start of the
            required coverage window.
        end (str): ISO date/datetime string marking the end of the
            required coverage window.

    Returns:
        list: IDs of the sensors that measure `pollutant_id` and have
            coverage spanning [start, end].

    Raises:
        StopIteration: If a station in `stations_list` has no sensor
            measuring `pollutant_id`.
    """
    # Select only the sensors which measure the given pollutant
    sensors_ids = [
        next(
            sensor.id
            for sensor in station.sensors
            if sensor.parameter.id == pollutant_id
        )
        for station in stations_list
    ]
    active_sensors = []
    for i, s_id in enumerate(sensors_ids):
        sens_result = openaq_client.sensors.get(sensors_id=s_id)
        if check_measurement_in_period(sens_result, start=start, end=end):
            active_sensors.append(s_id)
        if i % 50 == 0:
            print(f"Checking sensor number {i}")
    return active_sensors




def extract_sensor(row, seq):
    """Extracts the sensor ID from a nested list-dict format in the 'sensor' field of a station DataFrame."""
    for dic in row:
        if dic["id"] in seq:
            return dic["id"]
        else:
            continue
    return None
