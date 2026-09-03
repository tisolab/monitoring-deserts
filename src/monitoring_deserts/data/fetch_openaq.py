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
    """Check whether an OpenAQ station sensor has readings spanning
    the given range.

    Parameters:
        sensor: the result of an openaq.OpenAQ.Client.sensors.get()
                query.
        start, end: ISO date or datetime strings

    Returns False (rather than raising) when the sensor has no coverage
    information or the dates cannot be parsed.
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
    """Query the OpenAQ database for a pollutant with the given name."""
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
    """Pull the list of stations which measure the required pollutant."""
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
            ).results
            stations_list.extend(stl)
            if len(stl) < lim:
                break
            page += 1
            time.sleep(0.1)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch page {page} of the stations list from OpenAQ: {exc}"
        ) from exc
    return stations_list
