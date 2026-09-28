"""Helpers for the Phase F population and income layers (project plan, Sections 5.4-5.5).

Population (WorldPop, Section 5.4): `build_worldpop_url` builds the download
URL for one country's population-count GeoTIFF, `pull_worldpop` downloads one
such file, and `pull_worldpop_countries` does so for a list of countries.
`mosaic_rasters` then merges the per-country files into one region-wide raster.
Resampling that raster onto the analysis grid happens downstream, in the
notebook.

Two resolutions are supported, since the analysis grid (0.05 degrees, roughly
5-6 km) is far coarser than WorldPop's native 100 m: the full-resolution
product, and WorldPop's 1 km aggregated one, which is much smaller to download
and moves the same population totals onto the grid.

Income proxy (Meta's Relative Wealth Index, Section 5.5): `find_rwi_urls` looks
up the per-country CSV links on HDX, `pull_rwi` downloads them, `load_rwi`
reads them into one table, and `nearest_rwi` looks up the closest RWI point for
any set of locations, returning NaN when none is near enough.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio.errors
import requests
import rioxarray
from rioxarray.exceptions import NoDataInBounds, RioXarrayError
from rioxarray.merge import merge_arrays
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

WORLDPOP_BASE_URL = "https://data.worldpop.org/GIS/Population"

# {folder} is the resolution-specific top-level directory; {year}, {iso_upper}
# and {iso_lower} are filled in per request. The 1km pattern follows WorldPop's
# published layout but has not been checked against the live server -- if a
# 1km download 404s, correct it here.
WORLDPOP_PATTERNS = {
    "100m": "Global_2000_2020/{year}/{iso_upper}/{iso_lower}_ppp_{year}_UNadj.tif",
    "1km": "Global_2000_2020_1km_UNadj/{year}/{iso_upper}/{iso_lower}_ppp_{year}_1km_Aggregated_UNadj.tif",
}

WORLDPOP_YEARS = range(2000, 2021)

HDX_API_URL = "https://data.humdata.org/api/3/action/package_show"
HDX_RWI_DATASET = "relative-wealth-index"
# Per-country files are named like "lka_relative_wealth_index.csv". A few cover
# several countries in one file (India and Pakistan share "ind_pak_...").
RWI_FILE_PATTERN = re.compile(r"^([a-z]{3}(?:_[a-z]{3})*)_relative_wealth_index\.csv$")
RWI_COLUMNS = ["longitude", "latitude", "rwi", "error"]
EARTH_RADIUS_KM = 6371.0088


def build_worldpop_url(country, year=2020, resolution="100m"):
    """Build the WorldPop download URL for one country's population-count GeoTIFF.

    Uses the unconstrained, UN-adjusted product (Section 5.4), the version
    that is standardized across countries.

    Parameters:
        country (str): ISO 3166-1 alpha-3 country code, e.g. "IND", "PAK",
            "BGD". Case-insensitive.
        year (int): Population vintage, 2000-2020. Defaults to 2020.
        resolution (str): "100m" (native resolution) or "1km" (aggregated,
            much smaller files). Defaults to "100m".

    Returns:
        str: The full download URL.

    Raises:
        ValueError: If `country` isn't a 3-letter alphabetic code, `year`
            isn't an integer in 2000-2020, or `resolution` isn't supported.
    """
    if not isinstance(country, str) or len(country) != 3 or not country.isalpha():
        raise ValueError(
            f"country must be a 3-letter ISO 3166-1 alpha-3 code, got {country!r}"
        )
    if isinstance(year, bool) or not isinstance(year, int) or year not in WORLDPOP_YEARS:
        raise ValueError(
            f"year must be an integer in {WORLDPOP_YEARS.start}-"
            f"{WORLDPOP_YEARS.stop - 1}, got {year!r}"
        )
    if resolution not in WORLDPOP_PATTERNS:
        raise ValueError(
            f"resolution must be one of {sorted(WORLDPOP_PATTERNS)}, got {resolution!r}"
        )

    path = WORLDPOP_PATTERNS[resolution].format(
        year=year, iso_upper=country.upper(), iso_lower=country.lower()
    )
    return f"{WORLDPOP_BASE_URL}/{path}"


def _download_file(url, target, what, label, overwrite=False, progress=True):
    """Stream `url` to `target`, skipping if it already exists.

    Shared by the WorldPop and RWI downloads. The file is written to a
    temporary `.part` file and only renamed once complete, so an interrupted
    download never leaves a truncated file that looks finished.

    Parameters:
        url (str): File to download.
        target (Path): Final path. Its parent directory is created if missing.
        what (str): Description used in error messages, e.g. "WorldPop file
            for IND".
        label (str): Text shown on the progress bar and skip message.
        overwrite (bool): Download even if `target` exists. Defaults to False.
        progress (bool): Show a progress bar / skip message. Defaults to True.

    Returns:
        pathlib.Path: `target`.

    Raises:
        RuntimeError: On an HTTP error, connection problem, timeout or write
            failure. Any partial file is removed.
    """
    if target.exists() and not overwrite:
        if progress:
            tqdm.write(f"{label}: {target.name} already downloaded, skipping")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")

    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            # Content-Length can be missing, in which case tqdm just shows a running total
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(part, "wb") as f, tqdm(
                total=total,
                desc=label,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=not progress,
            ) as bar:
                # Small chunks: iter_content blocks until a full chunk arrives, so a large
                # chunk on this slow server leaves the bar frozen at 0% between updates
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    f.write(chunk)
                    bar.update(len(chunk))
    except requests.RequestException as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download {what} from {url}: {exc}"
            + (
                " (a 404 usually means the URL pattern or country code is wrong)"
                if isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and exc.response.status_code == 404
                else ""
            )
        ) from exc
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to write {part}: {exc}") from exc

    part.replace(target)
    return target


def pull_worldpop(
    country,
    data_dir,
    year=2020,
    resolution="100m",
    overwrite=False,
    progress=True,
    desc=None,
):
    """Download one country's WorldPop population-count GeoTIFF.

    The file is streamed to a temporary `.part` file and only renamed to its
    final name once the download completes, so an interrupted download never
    leaves a truncated `.tif` that looks complete. If the final file already
    exists it is left alone (unless `overwrite=True`), which makes re-running
    a multi-country pull resume where it stopped.

    Parameters:
        country (str): ISO 3166-1 alpha-3 country code, e.g. "IND".
        data_dir (str | Path): Directory to save into. Created if missing.
        year (int): Population vintage, 2000-2020. Defaults to 2020.
        resolution (str): "100m" or "1km". Defaults to "100m".
        overwrite (bool): Re-download even if the file already exists.
            Defaults to False.
        progress (bool): Show a byte-level progress bar for the download, and
            a message when an existing file is skipped. Defaults to True.
        desc (str | None): Label for the progress bar/message. Defaults to
            the uppercase country code.

    Returns:
        pathlib.Path: Path of the downloaded (or already present) file.

    Raises:
        ValueError: If `country`, `year` or `resolution` is invalid (see
            `build_worldpop_url`).
        RuntimeError: If the download fails -- an HTTP error (a 404 usually
            means the URL pattern or country code is wrong), a connection
            problem, or a timeout. Any partial file is removed.
    """
    url = build_worldpop_url(country, year=year, resolution=resolution)
    target = Path(data_dir) / url.rsplit("/", 1)[-1]
    return _download_file(
        url,
        target,
        what=f"WorldPop file for {country.upper()}",
        label=desc or country.upper(),
        overwrite=overwrite,
        progress=progress,
    )


def pull_worldpop_countries(
    countries, data_dir, year=2020, resolution="100m", overwrite=False, progress=True
):
    """Download WorldPop population-count GeoTIFFs for several countries.

    All country codes are validated up front, so a typo in the last one fails
    before any download starts. Files already on disk are skipped (unless
    `overwrite=True`), so after a failure part-way through, calling this again
    with the same arguments resumes with the countries not yet downloaded.

    Parameters:
        countries (list[str]): ISO 3166-1 alpha-3 country codes.
        data_dir (str | Path): Directory to save into. Created if missing.
        year (int): Population vintage, 2000-2020. Defaults to 2020.
        resolution (str): "100m" or "1km". Defaults to "100m".
        overwrite (bool): Re-download files that already exist. Defaults to
            False.
        progress (bool): Show a progress bar per file, labelled with a
            running country counter such as "[2/5] PAK". Defaults to True.

    Returns:
        dict[str, pathlib.Path]: Downloaded file path for each country,
            keyed by the uppercase country code.

    Raises:
        ValueError: If `countries` is empty, or any code, `year` or
            `resolution` is invalid.
        RuntimeError: If a download fails -- names the country that failed
            and how many countries had already been pulled before it.
    """
    if not countries:
        raise ValueError("countries is empty -- nothing to download.")

    for country in countries:
        build_worldpop_url(country, year=year, resolution=resolution)

    paths = {}
    for i, country in enumerate(countries, start=1):
        try:
            paths[country.upper()] = pull_worldpop(
                country,
                data_dir,
                year=year,
                resolution=resolution,
                overwrite=overwrite,
                progress=progress,
                desc=f"[{i}/{len(countries)}] {country.upper()}",
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Download failed for {country.upper()} ({len(paths)} of "
                f"{len(countries)} countries already pulled): {exc}"
            ) from exc

    return paths


def mosaic_rasters(paths, bbox=None):
    """Merge per-country WorldPop rasters into one region-wide population raster.

    Each file is opened with WorldPop's nodata value masked to NaN, optionally
    clipped to `bbox` *before* merging, then merged. Clipping first matters
    for memory: opened lazily, only the part inside `bbox` is ever read, which
    keeps a 100 m mosaic (billions of cells for a region this size) tractable.
    The 1 km product is small enough not to need it, but clipping does no harm.

    Parameters:
        paths (dict | list[str | Path]): Raster files to merge, e.g. the
            dict returned by `pull_worldpop_countries` (its values are used).
        bbox (tuple | None): `(min_lon, min_lat, max_lon, max_lat)` in
            degrees, this project's usual bounding-box order (unlike
            `fetch_era5`'s N/W/S/E). Defaults to None (no clipping).

    Returns:
        xarray.DataArray: Population per cell on a single `y`/`x` grid, named
            "population", NaN where no country has data. Units are people per
            cell of the source resolution -- still to be summed (not
            averaged) if resampled to a coarser grid.

    Raises:
        ValueError: If `paths` is empty, `bbox` is malformed, or the rasters
            don't all share one CRS.
        FileNotFoundError: If any file in `paths` doesn't exist.
        RuntimeError: If a file can't be read, has no data inside `bbox`, or
            the merge itself fails.
    """
    if isinstance(paths, dict):
        paths = list(paths.values())
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("paths is empty -- nothing to mosaic.")

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Raster file(s) not found: {missing}")

    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError(
                f"bbox must have exactly 4 elements (min_lon, min_lat, max_lon, "
                f"max_lat), got {bbox!r}"
            )
        min_lon, min_lat, max_lon, max_lat = bbox
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError(
                f"bbox must have min < max on both axes, got {bbox!r}"
            )

    rasters = []
    for path in paths:
        try:
            raster = rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)
            if bbox is not None:
                raster = raster.rio.clip_box(
                    minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat
                )
        except NoDataInBounds as exc:
            raise RuntimeError(f"{path.name} has no data inside bbox {bbox!r}") from exc
        except (rasterio.errors.RasterioError, RioXarrayError, OSError) as exc:
            raise RuntimeError(f"Failed to read raster {path}: {exc}") from exc
        rasters.append(raster)

    crs_found = {str(r.rio.crs) for r in rasters}
    if len(crs_found) > 1:
        raise ValueError(
            f"Rasters don't share one CRS, can't mosaic them: {sorted(crs_found)}"
        )

    try:
        merged = merge_arrays(rasters)
    except (rasterio.errors.RasterioError, RioXarrayError, ValueError) as exc:
        raise RuntimeError(f"Failed to merge rasters: {exc}") from exc

    merged.name = "population"
    return merged


def locate_grid_cells(longitude, latitude, grid_lons, grid_lats, cell_size):
    """Find the grid cell containing each point, by index arithmetic.

    For a regular grid this is exact and fast, with no spatial join, and it
    lets stations use the same cell values the model later sees on the grid.
    Cells are half-open (a point exactly on a shared edge belongs to the cell
    to its north/east).

    Parameters:
        longitude (array-like): Point longitudes in degrees.
        latitude (array-like): Point latitudes in degrees, same length.
        grid_lons (array-like): Ascending longitudes of the cell centres.
        grid_lats (array-like): Ascending latitudes of the cell centres.
        cell_size (float): Cell size in degrees.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: `(col, lat_idx)`, integer indices
            into `grid_lons` and `grid_lats` (both counted from the west/south,
            i.e. into the ascending arrays) of the containing cell for each
            point. To index a north-up raster use row `len(grid_lats) - 1 -
            lat_idx`.

    Raises:
        ValueError: If the inputs are malformed, `cell_size` isn't positive,
            or any point lies outside the grid.
    """
    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    lons = np.asarray(grid_lons, dtype=float)
    lats = np.asarray(grid_lats, dtype=float)
    if lon.ndim != 1 or lon.shape != lat.shape or lon.size == 0:
        raise ValueError("longitude and latitude must be non-empty 1-D arrays of the same length.")
    if np.isnan(lon).any() or np.isnan(lat).any():
        raise ValueError("longitude/latitude contain nulls.")
    if lons.ndim != 1 or lats.ndim != 1 or lons.size == 0 or lats.size == 0:
        raise ValueError("grid_lons and grid_lats must be non-empty 1-D arrays.")
    if not cell_size > 0:
        raise ValueError(f"cell_size must be positive, got {cell_size!r}")

    eps = 1e-9  # keep points on a cell edge from falling into the wrong cell through float error
    col = np.floor((lon - (lons[0] - cell_size / 2)) / cell_size + eps).astype(int)
    lat_idx = np.floor((lat - (lats[0] - cell_size / 2)) / cell_size + eps).astype(int)
    outside = (col < 0) | (col >= lons.size) | (lat_idx < 0) | (lat_idx >= lats.size)
    if outside.any():
        raise ValueError(f"{int(outside.sum())} point(s) lie outside the grid.")
    return col, lat_idx


def _check_country_codes(countries):
    """Return `countries` as uppercase codes, raising ValueError if any is invalid."""
    if not countries:
        raise ValueError("countries is empty -- nothing to look up.")
    for country in countries:
        if not isinstance(country, str) or len(country) != 3 or not country.isalpha():
            raise ValueError(
                f"country must be a 3-letter ISO 3166-1 alpha-3 code, got {country!r}"
            )
    return [country.upper() for country in countries]


def find_rwi_urls(countries):
    """Look up the HDX download URL of the Relative Wealth Index file for each country.

    Asks HDX's API for the RWI dataset's file list, since the download links
    contain opaque resource IDs that can't be built from the country code. Some
    files cover several countries (India and Pakistan share one), in which case
    those countries map to the same URL. Countries with no RWI file (RWI covers
    only about 93 low- and middle-income countries) are left out of the result
    with a warning rather than an error.

    Parameters:
        countries (list[str]): ISO 3166-1 alpha-3 country codes, e.g. "IND".

    Returns:
        dict[str, str]: Download URL for each requested country that has an
            RWI file, keyed by uppercase country code.

    Raises:
        ValueError: If `countries` is empty or contains an invalid code.
        RuntimeError: If HDX can't be reached or returns an unusable answer.
    """
    codes = _check_country_codes(countries)

    try:
        response = requests.get(
            HDX_API_URL, params={"id": HDX_RWI_DATASET}, timeout=30
        )
        response.raise_for_status()
        payload = response.json()
        resources = payload["result"]["resources"]
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not get the RWI file list from HDX: {exc}") from exc

    urls = {}
    for resource in resources:
        url = resource.get("url", "")
        match = RWI_FILE_PATTERN.match(url.rsplit("/", 1)[-1].lower())
        if match:
            for code in match.group(1).split("_"):
                urls.setdefault(code.upper(), url)

    found = {code: urls[code] for code in codes if code in urls}
    missing = sorted(set(codes) - set(found))
    if missing:
        warnings.warn(f"No RWI file on HDX for: {missing}")
    return found


def pull_rwi(countries, data_dir, overwrite=False, progress=True):
    """Download the Relative Wealth Index CSVs for several countries.

    Countries that share one file (India and Pakistan) download it once. Files
    already on disk are skipped (unless `overwrite=True`), so re-running after
    a failure resumes.

    Parameters:
        countries (list[str]): ISO 3166-1 alpha-3 country codes.
        data_dir (str | Path): Directory to save into. Created if missing.
        overwrite (bool): Re-download files that already exist. Defaults to
            False.
        progress (bool): Show a progress bar per file. Defaults to True.

    Returns:
        dict[str, pathlib.Path]: Local file for each requested country that
            has an RWI file, keyed by uppercase country code. Countries
            sharing a file share the same path.

    Raises:
        ValueError: If `countries` is empty or contains an invalid code.
        RuntimeError: If HDX can't be queried, none of the countries has an
            RWI file, or a download fails (names the file and how many were
            already pulled).
    """
    urls = find_rwi_urls(countries)
    if not urls:
        raise RuntimeError(f"None of {list(countries)} has an RWI file on HDX.")

    data_dir = Path(data_dir)
    unique_urls = list(dict.fromkeys(urls.values()))
    paths_by_url = {}
    for i, url in enumerate(unique_urls, start=1):
        name = url.rsplit("/", 1)[-1]
        try:
            paths_by_url[url] = _download_file(
                url,
                data_dir / name,
                what=f"RWI file {name}",
                label=f"[{i}/{len(unique_urls)}] {name}",
                overwrite=overwrite,
                progress=progress,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Download failed for {name} ({len(paths_by_url)} of "
                f"{len(unique_urls)} files already pulled): {exc}"
            ) from exc

    return {code: paths_by_url[url] for code, url in urls.items()}


def load_rwi(paths):
    """Read Relative Wealth Index CSVs into one table of wealth points.

    Parameters:
        paths (dict | list[str | Path]): CSV files, e.g. the dict returned by
            `pull_rwi`. Files listed more than once (India and Pakistan share
            one) are read once.

    Returns:
        pandas.DataFrame: Columns `longitude`, `latitude`, `rwi` (the wealth
            index, higher is wealthier) and `error` (its uncertainty), one row
            per point, with points repeated across neighbouring files dropped.

    Raises:
        ValueError: If `paths` is empty, or a file lacks the expected columns.
        FileNotFoundError: If a file doesn't exist.
        RuntimeError: If a file can't be parsed.
    """
    if isinstance(paths, dict):
        paths = list(paths.values())
    unique_paths = list(dict.fromkeys(Path(p) for p in paths))
    if not unique_paths:
        raise ValueError("paths is empty -- nothing to load.")

    missing = [str(p) for p in unique_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"RWI file(s) not found: {missing}")

    frames = []
    for path in unique_paths:
        try:
            df = pd.read_csv(path)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise RuntimeError(f"Failed to read RWI file {path}: {exc}") from exc
        absent = sorted(set(RWI_COLUMNS) - set(df.columns))
        if absent:
            raise ValueError(f"{path.name} is missing column(s) {absent}; has {list(df.columns)}")
        frames.append(df[RWI_COLUMNS])

    rwi = pd.concat(frames, ignore_index=True)
    incomplete = rwi[["longitude", "latitude", "rwi"]].isna().any(axis=1)
    if incomplete.any():
        warnings.warn(f"Dropping {int(incomplete.sum())} RWI row(s) with a missing coordinate or value")
        rwi = rwi[~incomplete]
    return rwi.drop_duplicates(subset=["longitude", "latitude"]).reset_index(drop=True)


def _to_unit_sphere(longitude, latitude):
    """Convert degrees to 3-D points on the unit sphere (so distances are true distances)."""
    lon, lat = np.radians(longitude), np.radians(latitude)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def nearest_rwi(longitude, latitude, rwi, max_distance_km=5.0):
    """Look up the nearest Relative Wealth Index point for each location.

    Distances are measured on the sphere (through 3-D unit-sphere coordinates),
    not in raw degrees, which stretch with latitude. A location with no RWI
    point within `max_distance_km` gets NaN instead of a far-away value: RWI
    only exists where people live and only for covered countries, so an
    unlimited nearest-neighbour search would silently copy wealth values onto
    empty land, ocean, and uncovered countries.

    Parameters:
        longitude (array-like): Longitudes in degrees, one per location.
        latitude (array-like): Latitudes in degrees, same length.
        rwi (pandas.DataFrame): RWI points, e.g. from `load_rwi`.
        max_distance_km (float): Farthest an RWI point may be and still be
            used. Defaults to 5.0 -- about twice RWI's own point spacing
            (~2.4 km) and roughly one analysis-grid cell (~5.5 km).

    Returns:
        pandas.DataFrame: One row per location, in input order, with columns
            `rwi`, `rwi_error` and `rwi_distance_km` (distance to the point
            used); all NaN where no point is within `max_distance_km`.

    Raises:
        ValueError: If the coordinates are empty, mismatched, null or out of
            range, `rwi` is empty or lacks columns, or `max_distance_km` isn't
            positive.
    """
    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    if lon.ndim != 1 or lon.shape != lat.shape:
        raise ValueError("longitude and latitude must be 1-D and the same length.")
    if lon.size == 0:
        raise ValueError("no locations given -- nothing to look up.")
    if np.isnan(lon).any() or np.isnan(lat).any() or (np.abs(lon) > 180).any() or (np.abs(lat) > 90).any():
        raise ValueError("longitude/latitude contain nulls or out-of-range values.")
    if not max_distance_km > 0:
        raise ValueError(f"max_distance_km must be positive, got {max_distance_km!r}")
    absent = sorted(set(RWI_COLUMNS) - set(rwi.columns))
    if absent:
        raise ValueError(f"rwi is missing column(s): {absent}")
    if rwi.empty:
        raise ValueError("rwi has no rows -- nothing to look up.")

    tree = cKDTree(_to_unit_sphere(rwi["longitude"].to_numpy(), rwi["latitude"].to_numpy()))
    max_chord = 2 * np.sin(max_distance_km / (2 * EARTH_RADIUS_KM))
    chord, idx = tree.query(_to_unit_sphere(lon, lat), distance_upper_bound=max_chord)

    found = np.isfinite(chord)
    safe_idx = np.where(found, idx, 0)  # cKDTree returns len(rwi) as the index when nothing is in range
    distance_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.where(found, chord, 0.0) / 2)
    return pd.DataFrame(
        {
            "rwi": np.where(found, rwi["rwi"].to_numpy()[safe_idx], np.nan),
            "rwi_error": np.where(found, rwi["error"].to_numpy()[safe_idx], np.nan),
            "rwi_distance_km": np.where(found, distance_km, np.nan),
        }
    )
