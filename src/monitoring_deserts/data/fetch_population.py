"""Helpers for the Phase F WorldPop population download (project plan, Section 5.4).

`build_worldpop_url` builds the download URL for one country's population-count
GeoTIFF, `pull_worldpop` downloads one such file, and `pull_worldpop_countries`
does so for a list of countries. `mosaic_rasters` then merges the per-country
files into one region-wide raster. Resampling that raster onto the analysis grid
happens downstream, in the notebook.

Two resolutions are supported, since the analysis grid (0.05 degrees, roughly
5-6 km) is far coarser than WorldPop's native 100 m: the full-resolution
product, and WorldPop's 1 km aggregated one, which is much smaller to download
and moves the same population totals onto the grid.
"""

from pathlib import Path

import rasterio.errors
import requests
import rioxarray
from rioxarray.exceptions import NoDataInBounds, RioXarrayError
from rioxarray.merge import merge_arrays
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
    data_dir = Path(data_dir)
    target = data_dir / url.rsplit("/", 1)[-1]

    label = desc or country.upper()

    if target.exists() and not overwrite:
        if progress:
            tqdm.write(f"{label}: {target.name} already downloaded, skipping")
        return target

    data_dir.mkdir(parents=True, exist_ok=True)
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
            f"Failed to download WorldPop file for {country.upper()} from {url}: {exc}"
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
