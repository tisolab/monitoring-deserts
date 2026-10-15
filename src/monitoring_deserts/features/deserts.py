"""Monitoring-desert statistics on the analysis grid (project plan, Phase M).

A cell is a "monitoring desert" at a given threshold when its nearest station is farther
than that many kilometres. The distances themselves come from
`monitoring_deserts.models.spatial_cv.nearest_distance_km`; this module handles what is
done with them:

1. the area of each grid cell, which shrinks towards the poles (a 0.05 degree cell is not
   a fixed number of square kilometres);
2. summarising, for several thresholds at once, how many cells, how much area and how
   many people lie in the desert, and how the predicted pollution there compares with the
   monitored cells, including a rank-based comparison that does not depend on the level
   of the predictions.

The threshold is a sensitivity parameter, not a fact: how far a station's reading can be
trusted to generalise varies with land use and pollutant, so results are reported over a
range of thresholds.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from monitoring_deserts.models.spatial_cv import EARTH_RADIUS_KM


def cell_area_km2(latitude, cell_size_deg):
    """Area of a longitude-latitude grid cell on a sphere, in square kilometres.

    The cell spans `cell_size_deg` in both directions and is centred on `latitude`. On a
    sphere the area is `R^2 * dlon * (sin(lat_north) - sin(lat_south))`, which is exact
    and shrinks with the cosine of the latitude.

    Parameters:
        latitude (array-like): Latitudes of the cell centres, in degrees.
        cell_size_deg (float): Cell size in degrees. Must be positive.

    Returns:
        numpy.ndarray: Area in km² for each cell, same order as `latitude`.

    Raises:
        ValueError: If `cell_size_deg` is not positive, or a cell would extend beyond a
            pole, or `latitude` contains nulls.
    """
    lat = np.asarray(latitude, dtype=float)
    if not cell_size_deg > 0:
        raise ValueError(f"cell_size_deg must be positive, got {cell_size_deg!r}")
    if np.isnan(lat).any():
        raise ValueError("latitude contains nulls.")
    half = cell_size_deg / 2
    if (np.abs(lat) + half > 90).any():
        raise ValueError("A cell extends beyond a pole.")

    d_lon = np.radians(cell_size_deg)
    return EARTH_RADIUS_KM**2 * d_lon * (np.sin(np.radians(lat + half)) - np.sin(np.radians(lat - half)))


def probability_higher(values_a, values_b):
    """Probability that a random value of `values_a` exceeds a random value of `values_b`.

    Ties count half. It equals the area under the ROC curve and the Mann-Whitney U
    statistic divided by `n_a * n_b`: 0.5 means the two groups are not ordered at all,
    values above 0.5 mean group A tends to be higher. It uses ranks only, so it does not
    depend on the scale of the values, which matters because predictions of high values
    are compressed (project plan, Phase J item 5).

    Parameters:
        values_a, values_b (array-like): The two groups. NaN values are dropped.

    Returns:
        float: The probability, or NaN if either group is empty after dropping NaN.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([a, b]))  # average ranks for ties
    u = ranks[: a.size].sum() - a.size * (a.size + 1) / 2
    return float(u / (a.size * b.size))


def desert_summary(distance_km, thresholds_km, area_km2, population=None, values=None):
    """Summarise the monitoring desert at several thresholds, one row per threshold.

    A cell is in the desert at threshold `t` if `distance_km > t`.

    Parameters:
        distance_km (array-like): Distance from each cell to the nearest station, in km.
        thresholds_km (sequence of float): Positive thresholds, in km, strictly increasing.
        area_km2 (array-like): Area of each cell in km² (see `cell_area_km2`).
        population (array-like | None): People in each cell. NaN means unknown and is left
            out of the sums; the number of desert cells with unknown population is reported.
        values (array-like | None): A per-cell value to compare between desert and
            monitored cells, e.g. the mean predicted PM2.5. NaN cells are left out.

    Returns:
        pd.DataFrame: One row per threshold with `threshold_km`, `n_desert_cells`,
            `share_cells`, `desert_area_km2`, `share_area`; if `population` is given,
            `desert_population`, `share_population` and `desert_cells_without_population`;
            if `values` is given, the mean in monitored and desert cells
            (`value_mean_monitored`, `value_mean_desert`), the same weighted by
            population when `population` is also given (`value_popweighted_*`), and
            `prob_desert_higher` (rank-based, see `probability_higher`).
            Statistics over an empty group are NaN.

    Raises:
        ValueError: If the inputs differ in length or are empty, `distance_km` has nulls
            or negatives, `area_km2` is not positive, `population` is negative, or the
            thresholds are not positive and strictly increasing.
    """
    distance = np.asarray(distance_km, dtype=float)
    area = np.asarray(area_km2, dtype=float)
    thresholds = list(thresholds_km)
    if distance.ndim != 1 or distance.size == 0 or area.shape != distance.shape:
        raise ValueError("distance_km and area_km2 must be non-empty 1-D arrays of the same length.")
    if np.isnan(distance).any() or (distance < 0).any():
        raise ValueError("distance_km must have no nulls and no negative values.")
    if not (area > 0).all():
        raise ValueError("area_km2 must be positive.")
    if len(thresholds) == 0 or not (np.diff(thresholds) > 0).all() or not thresholds[0] > 0:
        raise ValueError("thresholds_km must be positive and strictly increasing.")

    pop = None
    if population is not None:
        pop = np.asarray(population, dtype=float)
        if pop.shape != distance.shape:
            raise ValueError("population must have the same length as distance_km.")
        if (pop[~np.isnan(pop)] < 0).any():
            raise ValueError("population must not be negative.")
    val = None
    if values is not None:
        val = np.asarray(values, dtype=float)
        if val.shape != distance.shape:
            raise ValueError("values must have the same length as distance_km.")

    total_area = area.sum()
    total_pop = np.nansum(pop) if pop is not None else None
    rows = []
    for t in thresholds:
        desert = distance > t
        row = {
            "threshold_km": t,
            "n_desert_cells": int(desert.sum()),
            "share_cells": float(desert.mean()),
            "desert_area_km2": float(area[desert].sum()),
            "share_area": float(area[desert].sum() / total_area),
        }
        if pop is not None:
            row["desert_population"] = float(np.nansum(pop[desert]))
            row["share_population"] = float(row["desert_population"] / total_pop) if total_pop > 0 else float("nan")
            row["desert_cells_without_population"] = int(np.isnan(pop[desert]).sum())
        if val is not None:
            for label, mask in (("monitored", ~desert), ("desert", desert)):
                ok = mask & ~np.isnan(val)
                row[f"value_mean_{label}"] = float(val[ok].mean()) if ok.any() else float("nan")
                if pop is not None:
                    w = ok & ~np.isnan(pop) & (pop > 0)
                    row[f"value_popweighted_{label}"] = (
                        float(np.average(val[w], weights=pop[w])) if w.any() else float("nan")
                    )
            row["prob_desert_higher"] = probability_higher(val[desert], val[~desert])
        rows.append(row)
    return pd.DataFrame(rows)
