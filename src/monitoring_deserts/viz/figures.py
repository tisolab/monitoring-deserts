"""Figures for the README and the write-up.

Plotting functions that take plain arrays, DataFrames and numbers (read from the saved
tables and metadata by `scripts/make_report_assets.py`) and write a PNG. Colours follow one
small system: a single blue ramp for magnitude (PM2.5, distance thresholds), blue and orange
as the two categorical colours, and grey ink for text and axes. The categorical pair and the
ordinal ramp were checked with a colour-vision-deficiency validator; text is always drawn
in ink colours, never in a series colour.
"""

import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Polygon
from matplotlib.lines import Line2D

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # steps 100..700
MAP_RAMP = BLUE_RAMP[:5]  # steps 100-500: dark enough to read, light enough for dark hatching on top
THRESHOLD_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # four ordered thresholds, light to dark


def apply_style():
    """Set the matplotlib defaults used by every figure."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 9,
        "hatch.linewidth": 0.45, "hatch.color": INK,
    })


def sequential_cmap():
    """The blue ramp used for the maps, light to dark, as a matplotlib colormap."""
    return LinearSegmentedColormap.from_list("pm25_blue", MAP_RAMP)


def _title(fig, title, subtitle=None):
    """Draw a wrapped title and subtitle at the top left; return the figure fraction left below them."""
    width, height = fig.get_size_inches()
    wrapped = textwrap.fill(title, width=max(int(width * 6.4), 20))
    n_title = wrapped.count("\n") + 1
    fig.text(0.02, 1 - 0.12 / height, wrapped, ha="left", va="top", fontsize=14, fontweight="bold", color=INK, linespacing=1.15)
    block = 0.12 + 0.27 * n_title
    if subtitle:
        text = textwrap.fill(subtitle, width=max(int(width * 9.0), 20))
        fig.text(0.02, 1 - (block + 0.08) / height, text, ha="left", va="top", fontsize=9.5, color=INK_SECONDARY, linespacing=1.25)
        block += 0.08 + 0.20 * (text.count("\n") + 1)
    return 1 - (block + 0.30) / height


def _rounded_bar(ax, x0, x1, y, height, color, radius_x, radius_y):
    """A horizontal bar from `x0` to `x1` with a rounded data end (`x1`) and a square baseline end (`x0`).

    The corner radii are given in data units of each axis, so that they look equal on screen.
    """
    sign = 1.0 if x1 >= x0 else -1.0
    rx = min(radius_x, abs(x1 - x0))
    ry = min(radius_y, height / 2)
    top, bottom = y + height / 2, y - height / 2
    angles = np.linspace(0, np.pi / 2, 8)
    lower = [(x1 - sign * rx + sign * rx * np.cos(a), bottom + ry - ry * np.sin(a)) for a in angles]
    upper = [(x1 - sign * rx + sign * rx * np.sin(a), top - ry + ry * np.cos(a)) for a in angles]
    points = [(x0, bottom)] + lower[::-1] + upper[::-1] + [(x0, top)]
    ax.add_patch(Polygon(points, closed=True, fc=color, ec="none"))


def hero_map(pm25, distance_km, mask, stations, borders, extent, desert_km, path, title, subtitle):
    """Predicted PM2.5 over the study countries, the stations, and the area far from any station.

    Parameters:
        pm25, distance_km, mask (numpy.ndarray): 2-D north-up rasters on the analysis grid
            (row 0 is the northernmost row): predicted mean PM2.5, distance to the nearest
            station in km, and a boolean study-area mask. Cells outside the mask are not drawn.
        stations (pandas.DataFrame): `longitude` and `latitude` of the stations.
        borders (geopandas.GeoDataFrame): Country polygons, drawn as thin outlines.
        extent (sequence): `[west, east, south, north]` of the rasters, in degrees.
        desert_km (float): Distance beyond which a cell is hatched as a monitoring desert.
        path (str | Path): Output PNG.
        title, subtitle (str): Text above the map.
    """
    apply_style()
    shown = np.where(mask, pm25, np.nan)
    low, high = np.nanquantile(shown, [0.02, 0.98])
    west, east, south, north = extent
    fig = plt.figure(figsize=(8.6, 9.6))
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.76])
    ax.grid(False)
    image = ax.imshow(shown, extent=extent, cmap=sequential_cmap(), vmin=low, vmax=high, interpolation="nearest", zorder=1)

    lons = np.linspace(west, east, shown.shape[1])
    lats = np.linspace(north, south, shown.shape[0])
    far = np.where(mask & (distance_km > desert_km), 1.0, 0.0)
    ax.contourf(lons, lats, far, levels=[0.5, 1.5], colors="none", hatches=["///"], zorder=2)
    ax.contour(lons, lats, np.where(mask, distance_km, np.nan), levels=[desert_km], colors=INK_SECONDARY, linewidths=0.8, zorder=3)
    borders.boundary.plot(ax=ax, color=INK, linewidth=0.5, zorder=4)
    ax.scatter(stations["longitude"], stations["latitude"], s=16, color=ORANGE, edgecolor=SURFACE, linewidth=0.6, zorder=5)

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = fig.add_axes([0.20, 0.075, 0.38, 0.016])
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=8.5)
    cax.set_title("Mean predicted PM2.5 (µg/m³)", fontsize=9, color=INK_SECONDARY, loc="left", pad=4)
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor=SURFACE, markersize=7, label="monitoring station"),
        Patch(facecolor="none", edgecolor=INK, hatch="///", label=f"more than {desert_km:g} km from a station"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.64, 0.045), fontsize=9, labelcolor=INK_SECONDARY)
    _title(fig, title, subtitle)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def cv_ladder(metrics, forest_random_r2, path, title, subtitle):
    """Spatial cross-validated R² of each model, with the random-fold score of the forest marked.

    Parameters:
        metrics (dict): model name -> pooled spatial R², ordered from the weakest to the strongest
            baseline up to the feature-based models.
        forest_random_r2 (float): R² of the Random Forest scored on random folds.
        path (str | Path): Output PNG.
        title, subtitle (str): Text above the chart.
    """
    apply_style()
    names = list(metrics)
    feature_models = {"Ridge (same features)", "Random Forest", "XGBoost"}
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    top = _title(fig, title, subtitle)
    fig.subplots_adjust(left=0.30, right=0.93, top=top, bottom=0.14)
    ys = np.arange(len(names))[::-1]
    ax.set_xlim(-0.2, 0.7)
    for y, name in zip(ys, names):
        value = metrics[name]
        color = BLUE if name in feature_models else BASELINE
        _rounded_bar(ax, 0.0, value, y, 0.5, color, radius_x=0.009, radius_y=0.07)  # a negative value grows leftwards from zero
        ax.text(max(value, 0) + 0.012, y, f"{value:.2f}", va="center", ha="left", fontsize=9.5, color=INK)
    ax.axvline(0, color=BASELINE, linewidth=1.0, zorder=0)
    ax.axvline(forest_random_r2, color=ORANGE, linewidth=2.0, zorder=3)
    ax.text(forest_random_r2 - 0.012, ys[-1] - 0.62, "Random Forest on random folds", ha="right", va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("R² on held-out regions (pooled over 8 spatial folds)")
    ax.set_ylim(-0.9, len(names) - 0.4)
    handles = [Patch(color=BASELINE, label="uses no features"), Patch(color=BLUE, label="uses satellite and weather features"),
               Line2D([], [], color=ORANGE, linewidth=2.0, label="same forest, random folds")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.86, 1.0), fontsize=8.5, labelcolor=INK_SECONDARY)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def equity_share_by_group(shares, path, title, subtitle):
    """Share of each income group's people who live in the monitoring desert, one line per threshold.

    Parameters:
        shares (pandas.DataFrame): Rows are income groups from poorest to richest, columns are the
            thresholds in km, values are shares between 0 and 1 (columns ordered from the smallest
            threshold).
        path (str | Path): Output PNG.
        title, subtitle (str): Text above the chart.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    top = _title(fig, title, subtitle)
    fig.subplots_adjust(left=0.13, right=0.84, top=top, bottom=0.17)
    x = np.arange(len(shares))
    for color, column in zip(THRESHOLD_RAMP, shares.columns):
        y = shares[column].to_numpy() * 100
        ax.plot(x, y, color=color, linewidth=2.0, solid_capstyle="round")
        ax.scatter(x, y, s=36, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(x[-1] + 0.12, y[-1], f"> {int(column)} km", va="center", ha="left", fontsize=9.5, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(["poorest\nfifth"] + [""] * (len(x) - 2) + ["richest\nfifth"])
    ax.set_xlim(-0.2, len(x) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Income group (Relative Wealth Index, equal numbers of people)")
    ax.set_ylabel("Share of the group's people living\nfarther than the threshold from a station")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def equity_gap_and_correlations(gaps, correlations, path, title, subtitle):
    """Left: poorest-minus-richest gap by threshold for two income proxies. Right: rank correlations.

    Parameters:
        gaps (pandas.DataFrame): Columns `proxy`, `threshold_km`, `estimate`, `lower`, `upper`, with the
            gap and its interval as shares between -1 and 1 (two proxies, named "RWI" and
            "night light per person").
        correlations (pandas.DataFrame): Columns `label`, `estimate`, `lower`, `upper`, ordered as they
            are to be drawn from top to bottom.
        path (str | Path): Output PNG.
        title, subtitle (str): Text above the figure.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    top = _title(fig, title, subtitle)
    fig.subplots_adjust(left=0.075, right=0.97, top=top - 0.06, bottom=0.19, wspace=0.62)

    ax = axes[0]
    thresholds = sorted(gaps["threshold_km"].unique())
    rows = np.arange(len(thresholds))[::-1]
    for offset, proxy, color in ((0.14, "RWI", BLUE), (-0.14, "night light per person", ORANGE)):
        part = gaps[gaps["proxy"] == proxy].set_index("threshold_km").loc[thresholds]
        y = rows + offset
        ax.hlines(y, part["lower"] * 100, part["upper"] * 100, color=color, linewidth=2.0)
        ax.scatter(part["estimate"] * 100, y, s=48, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=proxy)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1.0)
    ax.set_yticks(rows)
    ax.set_yticklabels([f"> {t} km" for t in thresholds])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Poorest minus richest fifth: percentage points more\nof their people living in the desert (95% interval)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=9, labelcolor=INK_SECONDARY, columnspacing=1.5, handletextpad=0.4)

    ax = axes[1]
    n = len(correlations)
    rows = np.arange(n)[::-1]
    ax.hlines(rows, correlations["lower"], correlations["upper"], color=BLUE, linewidth=2.0)
    ax.scatter(correlations["estimate"], rows, s=48, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1.0)
    ax.set_yticks(rows)
    ax.set_yticklabels([textwrap.fill(label, 22) for label in correlations["label"]])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Rank correlation of distance to the nearest station\nwith the Relative Wealth Index (95% interval)")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def uncertainty_panels(coverage_by_fold, level, out_of_range_share, path, title, subtitle):
    """Left: share of held-out values inside the uncertainty band, per spatial fold. Right: how unlike the stations the grid is.

    Parameters:
        coverage_by_fold (dict): fold number -> share (0-1) of held-out observations inside the band.
        level (float): Nominal coverage of the band, e.g. 0.9; drawn as a reference line.
        out_of_range_share (dict): label -> share (0-1) of predicted cell-months, in the order
            "no feature outside", "1 or 2 features outside", "3 or more features outside".
        path (str | Path): Output PNG.
        title, subtitle (str): Text above the figure.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    top = _title(fig, title, subtitle)
    fig.subplots_adjust(left=0.07, right=0.97, top=top, bottom=0.17, wspace=0.35)

    ax = axes[0]
    folds = list(coverage_by_fold)
    values = np.array([coverage_by_fold[f] for f in folds]) * 100
    ax.bar(np.arange(len(folds)), values, width=0.4, color=BLUE)
    ax.axhline(level * 100, color=ORANGE, linewidth=2.0)
    ax.text(-0.45, level * 100 + 1.8, f"intended {level:.0%}", ha="left", va="bottom", fontsize=9, color=INK_SECONDARY)
    worst = int(np.argmin(values))
    ax.text(worst, values[worst] + 1.5, f"{values[worst]:.0f}%", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(np.arange(len(folds)))
    ax.set_xticklabels([str(int(f)) for f in folds])
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Spatial fold (held-out region)")
    ax.set_ylabel("Observations inside the band")

    ax = axes[1]
    labels = list(out_of_range_share)
    shares = np.array([out_of_range_share[k] for k in labels]) * 100
    ramp = [THRESHOLD_RAMP[0], THRESHOLD_RAMP[1], THRESHOLD_RAMP[3]]
    left = 0.0
    for label, share, color in zip(labels, shares, ramp):
        ax.barh(0, share, left=left, height=0.3, color=color, edgecolor=SURFACE, linewidth=2)
        left += share
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.2, 0.9)
    ax.set_yticks([])
    ax.grid(visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)])
    ax.set_xlabel("Share of predicted cell-months")
    handles = [Patch(color=c, label=f"{l}: {s:.0f}%") for l, s, c in zip(labels, shares, ramp)]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=9, labelcolor=INK_SECONDARY)
    fig.savefig(path, dpi=200)
    plt.close(fig)
