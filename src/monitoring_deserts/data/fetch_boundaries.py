"""Country boundaries and a land mask for the analysis grid (used for maps and report figures).

The analysis grid covers the whole bounding box of the study region, so it includes sea and
land outside the countries the analysis is about. For presentation, the maps are restricted
to the countries with population data. This module handles two things:

1. downloading the Natural Earth country boundaries (public domain) once, and reading the
   countries wanted;
2. turning those polygons into a boolean raster on the analysis grid, keeping the cells whose
   centre lies inside.

Natural Earth draws borders following its own conventions for disputed areas; the mask
follows them and takes no position on them.
"""

from pathlib import Path

import numpy as np
import requests
import geopandas as gpd
import shapely

NATURAL_EARTH_URL = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
ISO_COLUMN = "ADM0_A3"  # Natural Earth's three-letter country code, unlike ISO_A3 never "-99"
DEFAULT_COUNTRIES = ("IND", "PAK", "BGD", "NPL", "LKA")  # the countries with WorldPop data in this project


def download_country_boundaries(data_dir, url=NATURAL_EARTH_URL, overwrite=False, timeout=120):
    """Download the Natural Earth 1:50m country boundaries (a zipped shapefile).

    Parameters:
        data_dir (str | Path): Folder to save into. Created if missing.
        url (str): Source of the zip file. Defaults to Natural Earth's 50m admin-0 countries.
        overwrite (bool): Download again even if the file exists. Defaults to False, so a
            second call does nothing.
        timeout (int): Seconds to wait for the server. Defaults to 120.

    Returns:
        pathlib.Path: Path of the zip file.

    Raises:
        RuntimeError: If the download fails.
    """
    target = Path(data_dir) / url.rsplit("/", 1)[-1]
    if target.exists() and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc
    target.write_bytes(response.content)
    return target


def load_countries(path, iso_codes=DEFAULT_COUNTRIES):
    """Read the boundaries of the wanted countries.

    Parameters:
        path (str | Path): The zipped shapefile from `download_country_boundaries`, or any
            file geopandas can read that has an `ADM0_A3` column.
        iso_codes (sequence of str): Three-letter codes of the countries to keep.

    Returns:
        geopandas.GeoDataFrame: One row per country with its geometry in EPSG:4326.

    Raises:
        ValueError: If `iso_codes` is empty, or a code is not in the file.
    """
    codes = [str(code).upper() for code in iso_codes]
    if not codes:
        raise ValueError("iso_codes is empty.")
    countries = gpd.read_file(path)
    if ISO_COLUMN not in countries.columns:
        raise ValueError(f"{path} has no '{ISO_COLUMN}' column.")
    missing = sorted(set(codes) - set(countries[ISO_COLUMN]))
    if missing:
        raise ValueError(f"Countries not found in the boundaries file: {missing}")
    return countries[countries[ISO_COLUMN].isin(codes)].to_crs("EPSG:4326").reset_index(drop=True)


def country_mask(grid_lons, grid_lats, countries):
    """Boolean raster of the grid cells whose centre lies inside the countries.

    Parameters:
        grid_lons, grid_lats (array-like): Ascending longitudes and latitudes of the cell
            centres, in degrees (the grid axes used throughout the project).
        countries (geopandas.GeoDataFrame | shapely geometry): The polygons to keep.

    Returns:
        numpy.ndarray: Boolean array of shape `(len(grid_lats), len(grid_lons))` indexed
            `[latitude index, longitude index]`, with ascending latitude (row 0 is the
            SOUTHERNMOST row, unlike a north-up raster).

    Raises:
        ValueError: If an axis is empty, not 1-D or not strictly ascending.
    """
    lons = np.asarray(grid_lons, dtype=float)
    lats = np.asarray(grid_lats, dtype=float)
    for axis in (lons, lats):
        if axis.ndim != 1 or axis.size == 0 or (np.diff(axis) <= 0).any():
            raise ValueError("grid_lons and grid_lats must be non-empty, 1-D and strictly ascending.")
    union = countries.geometry.union_all() if hasattr(countries, "geometry") else countries
    shapely.prepare(union)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return shapely.contains_xy(union, lon_grid, lat_grid)
