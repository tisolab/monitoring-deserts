"""Statistics for the equity analysis (project plan, Phase N).

The question is whether the gaps in monitoring are concentrated in lower-income areas.
Every cell of the grid has a population, an income proxy and a distance to the nearest
station, and the statistics here compare groups of people, not groups of cells. This
module handles four things:

1. income groups with equal numbers of *people* (population-weighted quantiles), so
   "poorest fifth" means a fifth of the population, not a fifth of the cells;
2. a table of how many people of each group live in the desert and how polluted their
   cells are predicted to be, in the desert and outside it;
3. a partial rank correlation, to ask whether distance to a station and income are
   related once the population density (a stand-in for urban against rural) is accounted
   for;
4. a bootstrap over spatial blocks for any statistic. Neighbouring cells are strongly
   correlated, so a bootstrap over single cells, or a textbook p-value, would make a result
   look far more certain than it is; resampling whole blocks of the map keeps that
   correlation in the resampled data.

Income proxies are proxies, not income, and the predicted PM2.5 carries the uncertainty
of notebook 11; neither is corrected here.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def weighted_mean(values, weights):
    """Weighted mean of `values`; NaN if the weights sum to zero.

    Parameters:
        values (array-like): Values, no NaN.
        weights (array-like): Non-negative weights, same length.

    Raises:
        ValueError: If the lengths differ, the inputs are empty, or values contain NaN
            or a weight is negative or NaN.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.ndim != 1 or v.shape != w.shape or v.size == 0:
        raise ValueError("values and weights must be non-empty 1-D arrays of the same length.")
    if np.isnan(v).any() or np.isnan(w).any() or (w < 0).any():
        raise ValueError("values must have no NaN and weights must be non-negative numbers.")
    total = w.sum()
    return float((v * w).sum() / total) if total > 0 else float("nan")


def weighted_quantile_groups(values, weights, n_groups=5):
    """Assign each row to a group holding an equal share of the total weight.

    Rows are ordered by `values` and cut where the cumulative weight reaches 1/n, 2/n, ...
    of the total. With population as the weight, each group holds the same number of
    people, whatever the number of cells. Equal values always go in the same group, so
    group weights can differ slightly when there are many ties.

    Parameters:
        values (array-like): The quantity to rank by, e.g. the income proxy. NaN rows get
            no group.
        weights (array-like): Non-negative weights (e.g. population). Rows with a zero or
            NaN weight get no group.
        n_groups (int): Number of groups, at least 2. Defaults to 5 (quintiles).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: `(labels, cutoffs)`. `labels` is a float array
            with 0 for the lowest group up to `n_groups - 1`, NaN where there is no group.
            `cutoffs` holds the `n_groups - 1` upper edges of the lower groups.

    Raises:
        ValueError: If the inputs differ in length, `n_groups` is not an integer of at
            least 2, or fewer than `n_groups` rows have a value and a positive weight.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.ndim != 1 or v.shape != w.shape:
        raise ValueError("values and weights must be 1-D arrays of the same length.")
    if not (isinstance(n_groups, int) and n_groups >= 2):
        raise ValueError(f"n_groups must be an integer of at least 2, got {n_groups!r}")
    usable = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
    if usable.sum() < n_groups:
        raise ValueError(f"Need at least {n_groups} rows with a value and a positive weight, got {int(usable.sum())}.")

    order = np.argsort(v[usable], kind="stable")
    sorted_values = v[usable][order]
    share = np.cumsum(w[usable][order]) / w[usable].sum()
    targets = np.arange(1, n_groups) / n_groups
    cutoffs = sorted_values[np.minimum(np.searchsorted(share, targets, side="left"), len(share) - 1)]

    labels = np.full(v.shape, np.nan)
    labels[usable] = np.searchsorted(cutoffs, v[usable], side="left")
    return labels, cutoffs


def group_status_table(groups, in_desert, population, values=None):
    """People per group, how many are in the desert, and the mean `values` for each side.

    Parameters:
        groups (array-like): Group label of each cell (e.g. income quintile); NaN cells
            are left out.
        in_desert (array-like of bool): Whether each cell is in the desert.
        population (array-like): People in each cell; NaN or negative is an error, so
            select cells with population first.
        values (array-like | None): A per-cell value (e.g. predicted PM2.5) whose
            population-weighted mean is reported for the monitored and the desert cells.

    Returns:
        pd.DataFrame: One row per group with `population`, `desert_population`,
            `share_in_desert` and, if `values` is given, `mean_value_monitored` and
            `mean_value_desert` (NaN if a side has no people).

    Raises:
        ValueError: If the inputs differ in length or `population` has NaN or negatives.
    """
    g = np.asarray(groups, dtype=float)
    d = np.asarray(in_desert, dtype=bool)
    p = np.asarray(population, dtype=float)
    if g.ndim != 1 or not (g.shape == d.shape == p.shape):
        raise ValueError("groups, in_desert and population must be 1-D arrays of the same length.")
    if np.isnan(p).any() or (p < 0).any():
        raise ValueError("population must have no NaN and no negative values.")
    val = None
    if values is not None:
        val = np.asarray(values, dtype=float)
        if val.shape != g.shape:
            raise ValueError("values must have the same length as groups.")

    rows = {}
    for label in np.unique(g[~np.isnan(g)]):
        in_group = g == label
        row = {
            "population": float(p[in_group].sum()),
            "desert_population": float(p[in_group & d].sum()),
        }
        row["share_in_desert"] = row["desert_population"] / row["population"] if row["population"] > 0 else float("nan")
        if val is not None:
            for name, mask in (("monitored", in_group & ~d), ("desert", in_group & d)):
                ok = mask & ~np.isnan(val)
                row[f"mean_value_{name}"] = weighted_mean(val[ok], p[ok]) if ok.any() else float("nan")
        rows[int(label)] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "group"
    return table


def partial_spearman(x, y, controls=None):
    """Rank correlation of `x` and `y` after removing what `controls` explain of both.

    Everything is converted to ranks, `x` and `y` are each regressed on the (ranked)
    controls with an intercept, and the Pearson correlation of the two residuals is
    returned. With no controls this is the ordinary Spearman correlation. A control that
    drives both variables (population density drives both distance to a station and the
    income proxy) can create or hide an association, and this removes its linear rank
    effect; it cannot remove a nonlinear or missed confounder.

    Parameters:
        x, y (array-like): The two variables, same length, no NaN.
        controls (array-like | None): A 1-D array or a 2-D array with one column per
            control, same number of rows.

    Returns:
        float: The partial rank correlation, between -1 and 1; NaN if a residual has no
            spread.

    Raises:
        ValueError: If the lengths differ, any input has NaN, or there are fewer rows
            than controls plus 3.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.ndim != 1 or xa.shape != ya.shape:
        raise ValueError("x and y must be 1-D arrays of the same length.")
    if controls is None:
        c = np.empty((xa.size, 0))
    else:
        c = np.asarray(controls, dtype=float)
        c = c[:, None] if c.ndim == 1 else c
        if c.ndim != 2 or c.shape[0] != xa.size:
            raise ValueError("controls must have one row per observation.")
    if np.isnan(xa).any() or np.isnan(ya).any() or np.isnan(c).any():
        raise ValueError("Inputs must not contain NaN.")
    if xa.size < c.shape[1] + 3:
        raise ValueError("Too few rows for the number of controls.")

    rx, ry = rankdata(xa), rankdata(ya)
    design = np.column_stack([np.ones(xa.size)] + [rankdata(c[:, j]) for j in range(c.shape[1])])
    res_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    res_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if res_x.std() < 1e-8 or res_y.std() < 1e-8:  # a residual that is float noise has no spread
        return float("nan")
    return float(np.corrcoef(res_x, res_y)[0, 1])


def block_bootstrap(statistic, block_ids, n_boot=500, level=0.95, random_state=42):
    """Confidence interval of a statistic from resampling whole spatial blocks.

    Each replicate draws as many blocks as there are, with replacement, stacks their
    rows and recomputes the statistic. Because rows of a block stay together, the strong
    correlation between neighbouring cells is kept in every replicate, and the interval
    reflects how much the result depends on which parts of the map are in the sample.

    Parameters:
        statistic (callable): Takes an integer array of row indices (possibly with
            repeats) and returns a float.
        block_ids (array-like): Block label of every row (e.g. the 3-degree tile of a
            cell), at least 5 distinct blocks.
        n_boot (int): Number of replicates. Defaults to 500.
        level (float): Coverage of the interval, strictly between 0 and 1. Defaults to 0.95.
        random_state (int): Seed. Defaults to 42.

    Returns:
        dict: `estimate` (statistic on all rows), `lower` and `upper` (percentile
            interval), `std` (of the replicates), `n_blocks`, `n_boot` and `n_failed`
            (replicates whose statistic was NaN, which are dropped).

    Raises:
        ValueError: If there are fewer than 5 blocks, `n_boot` is below 10 or `level` is
            not strictly between 0 and 1.
    """
    ids = np.asarray(block_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("block_ids must be a non-empty 1-D array.")
    if not (isinstance(n_boot, int) and n_boot >= 10):
        raise ValueError(f"n_boot must be an integer of at least 10, got {n_boot!r}")
    if not 0 < level < 1:
        raise ValueError(f"level must be strictly between 0 and 1, got {level!r}")

    order = np.argsort(ids, kind="stable")
    _, starts = np.unique(ids[order], return_index=True)
    rows_of_block = np.split(order, starts[1:])
    n_blocks = len(rows_of_block)
    if n_blocks < 5:
        raise ValueError(f"Need at least 5 blocks to resample, got {n_blocks}.")

    rng = np.random.default_rng(random_state)
    replicates = []
    for _ in range(n_boot):
        drawn = rng.integers(0, n_blocks, size=n_blocks)
        value = statistic(np.concatenate([rows_of_block[b] for b in drawn]))
        replicates.append(value)
    replicates = np.asarray(replicates, dtype=float)
    ok = replicates[~np.isnan(replicates)]
    lower, upper = (np.quantile(ok, [(1 - level) / 2, (1 + level) / 2]) if ok.size else (np.nan, np.nan))
    return {
        "estimate": float(statistic(np.arange(ids.size))),
        "lower": float(lower),
        "upper": float(upper),
        "std": float(ok.std(ddof=1)) if ok.size > 1 else float("nan"),
        "n_blocks": int(n_blocks),
        "n_boot": int(n_boot),
        "n_failed": int(n_boot - ok.size),
    }
