"""An interactive web map of the results (Leaflet, through folium).

One self-contained HTML page: the predicted PM2.5 surface and the area far from any station
as image layers over a light base map, the stations, and the zones where a new monitor would
cover the most exposure. The page is meant to be served as it is (for example from GitHub
Pages): it needs a network connection for the base map tiles and the Leaflet scripts, but the
data is embedded.

Leaflet stretches an image linearly between its corner points in Web Mercator, while the grid
is regular in longitude and latitude, so the rasters are first resampled to rows that are
regular in Mercator y. Without that, features would be off by up to about 50 km at mid latitudes.
"""

import base64
import io

import branca.colormap as bcm
import folium
import matplotlib.image as mpimg
import numpy as np
from matplotlib.colors import to_rgba

from monitoring_deserts.viz.figures import MAP_RAMP, ORANGE, sequential_cmap

MAX_MERCATOR_LAT = 85.0


def _mercator_y(lat_deg):
    return np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))


def _inverse_mercator_y(y):
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)


def to_web_mercator_rows(raster, north, south):
    """Resample a north-up raster that is regular in latitude to rows regular in Web Mercator y.

    Parameters:
        raster (numpy.ndarray): 2-D array, row 0 the northernmost row, rows evenly spaced in
            latitude between `north` and `south` (the outer edges of the first and last row).
        north, south (float): Latitudes of the top and bottom edges of the raster, in degrees.

    Returns:
        numpy.ndarray: Array of the same shape, nearest-neighbour resampled.

    Raises:
        ValueError: If `raster` is not 2-D, or `north` is not above `south`, or a latitude is
            beyond +/- 85 degrees.
    """
    raster = np.asarray(raster)
    if raster.ndim != 2 or raster.shape[0] < 2:
        raise ValueError("raster must be 2-D with at least two rows.")
    if not north > south or max(abs(north), abs(south)) > MAX_MERCATOR_LAT:
        raise ValueError("Need north > south, both within 85 degrees of the equator.")
    n_rows = raster.shape[0]
    y_top, y_bottom = _mercator_y(north), _mercator_y(south)
    y_rows = y_top + (np.arange(n_rows) + 0.5) / n_rows * (y_bottom - y_top)
    lat_rows = _inverse_mercator_y(y_rows)
    source_rows = np.clip(np.floor((north - lat_rows) / (north - south) * n_rows).astype(int), 0, n_rows - 1)
    return raster[source_rows]


def _png_data_uri(rgba):
    buffer = io.BytesIO()
    mpimg.imsave(buffer, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_map(pm25, far_from_station, extent, stations, zones, path, desert_km, vmin, vmax, title, subtitle, repo_url=None):
    """Write the interactive map to an HTML file.

    Parameters:
        pm25 (numpy.ndarray): 2-D north-up raster of the mean predicted PM2.5, NaN where nothing is
            to be drawn (outside the study countries).
        far_from_station (numpy.ndarray): Boolean raster of the same shape, True in the monitoring desert.
        extent (sequence): `[west, east, south, north]` of the rasters, in degrees.
        stations (pandas.DataFrame): `station_id`, `longitude`, `latitude`, `n_months`, `mean_pm25`.
        zones (pandas.DataFrame): Candidate monitor zones, best first, with `rank`, `lon`, `lat`,
            `population`, `mean_pm25`, `mean_distance_km`.
        path (str | Path): Output HTML file.
        desert_km (float): The desert threshold, used in labels.
        vmin, vmax (float): Colour scale limits of the PM2.5 layer.
        title, subtitle (str): Text of the box at the top left of the page.
        repo_url (str | None): Link shown in the box, if given.

    Raises:
        ValueError: If the two rasters differ in shape or `vmax` is not above `vmin`.
    """
    pm25 = np.asarray(pm25, dtype=float)
    far = np.asarray(far_from_station, dtype=bool)
    if pm25.shape != far.shape:
        raise ValueError("pm25 and far_from_station must have the same shape.")
    if not vmax > vmin:
        raise ValueError("vmax must be above vmin.")
    west, east, south, north = extent

    pm25_mercator = to_web_mercator_rows(pm25, north, south)
    far_mercator = to_web_mercator_rows(far, north, south)
    cmap = sequential_cmap()
    rgba = cmap(np.clip((pm25_mercator - vmin) / (vmax - vmin), 0, 1))
    rgba[np.isnan(pm25_mercator)] = (0, 0, 0, 0)
    veil = np.zeros(far_mercator.shape + (4,))
    veil[far_mercator & ~np.isnan(pm25_mercator)] = to_rgba("#0b0b0b", 0.30)

    centre = [(south + north) / 2, (west + east) / 2]
    fmap = folium.Map(location=centre, zoom_start=5, min_zoom=4, tiles="CartoDB positron", control_scale=True)
    bounds = [[south, west], [north, east]]
    folium.raster_layers.ImageOverlay(
        image=_png_data_uri(rgba), bounds=bounds, opacity=0.88, name="Predicted PM2.5 (mean, µg/m³)", interactive=False, zindex=2,
    ).add_to(fmap)
    folium.raster_layers.ImageOverlay(
        image=_png_data_uri(veil), bounds=bounds, opacity=1.0, name=f"Shaded: more than {desert_km:g} km from a station",
        interactive=False, zindex=3, show=True,
    ).add_to(fmap)

    station_layer = folium.FeatureGroup(name=f"Monitoring stations ({len(stations)})", show=True)
    for station in stations.itertuples():
        folium.CircleMarker(
            [station.latitude, station.longitude], radius=3.5, color="#fcfcfb", weight=1, fill=True, fill_color=ORANGE, fill_opacity=1.0,
            popup=folium.Popup(f"<b>Station {station.station_id}</b><br>{station.n_months} months of PM2.5<br>mean observed {station.mean_pm25:.0f} µg/m³", max_width=220),
        ).add_to(station_layer)
    station_layer.add_to(fmap)

    zone_layer = folium.FeatureGroup(name=f"Top {len(zones)} zones for a new monitor", show=True)
    max_population = float(zones["population"].max())
    for zone in zones.itertuples():
        radius = 5 + 9 * np.sqrt(zone.population / max_population)
        folium.CircleMarker(
            [zone.lat, zone.lon], radius=float(radius), color="#0d366b", weight=2, fill=True, fill_color="#fcfcfb", fill_opacity=0.35,
            tooltip=(f"Rank {int(zone.rank)}: about {zone.population / 1e6:.1f} million people, "
                     f"mean predicted PM2.5 {zone.mean_pm25:.0f} µg/m³, {zone.mean_distance_km:.0f} km from the nearest station"),
        ).add_to(zone_layer)
    zone_layer.add_to(fmap)

    legend = bcm.LinearColormap(MAP_RAMP, vmin=vmin, vmax=vmax, caption="Mean predicted PM2.5 (µg/m³)")
    legend.add_to(fmap)
    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)

    link = f'<br><a href="{repo_url}" target="_blank" rel="noopener">Code and write-up</a>' if repo_url else ""
    box = f"""
    <div style="position: fixed; top: 10px; left: 55px; z-index: 9999; max-width: 340px; padding: 10px 14px;
                background: rgba(252,252,251,0.94); border-radius: 6px; border: 1px solid #e1e0d9;
                font: 13px/1.4 system-ui, -apple-system, 'Segoe UI', sans-serif; color: #0b0b0b;">
      <div style="font-weight: 700; font-size: 15px;">{title}</div>
      <div style="color: #52514e; margin-top: 3px;">{subtitle}</div>{link}
    </div>"""
    fmap.get_root().html.add_child(folium.Element(box))
    fmap.get_root().header.add_child(folium.Element(f"<title>{title}</title>"))
    fmap.save(str(path))
