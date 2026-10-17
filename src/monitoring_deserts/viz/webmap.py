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
import json

import branca.colormap as bcm
import folium
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
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


def _png_data_uri(rgba, colours=None):
    """A PNG data URI of an RGBA float array. With `colours`, the image is reduced to a palette of that size
    (the colour ramp is smooth, so little is lost) to keep the many month frames small."""
    buffer = io.BytesIO()
    if colours:
        image = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA")
        image.quantize(colors=colours, method=Image.Quantize.FASTOCTREE).save(buffer, format="PNG", optimize=True)
    else:
        mpimg.imsave(buffer, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


MONTH_CONTROL_CSS = """
<style>
  .month-control { background: rgba(252,252,251,0.96); border: 1px solid #e1e0d9; border-radius: 6px; padding: 8px 10px;
                   width: 270px; font: 13px/1.4 system-ui, -apple-system, 'Segoe UI', sans-serif; color: #0b0b0b; margin-bottom: 26px !important; }
  .month-control .mc-title { font-weight: 700; margin-bottom: 4px; }
  .month-control .mc-row { display: flex; gap: 6px; align-items: center; }
  .month-control select { flex: 1; padding: 4px; font: inherit; }
  .month-control button { padding: 3px 9px; font: inherit; cursor: pointer; }
  .month-control input[type=range] { width: 100%; margin: 8px 0 2px; }
  .month-control .mc-note { color: #52514e; font-size: 12px; }
</style>"""

MONTH_CONTROL_JS = """
// Runs after the page has loaded: folium writes the map's own script after this one, so the map does not exist before then.
window.addEventListener("load", function () {
  var map = __MAP__, overlay = __OVERLAY__, frames = __FRAMES__, last = frames.length - 1;
  var average = frames[0].label;
  var control = L.control({position: "bottomleft"});
  control.onAdd = function () {
    var div = L.DomUtil.create("div", "month-control");
    div.innerHTML = '<div class="mc-title">Prediction month</div>' +
      '<div class="mc-row"><button type="button" id="mc-prev" aria-label="Previous month">&#9664;</button>' +
      '<select id="mc-select" aria-label="Prediction month"></select>' +
      '<button type="button" id="mc-next" aria-label="Next month">&#9654;</button></div>' +
      '<input type="range" id="mc-range" min="1" max="' + last + '" step="1" value="' + last + '" aria-label="Prediction month slider">' +
      '<div class="mc-note" id="mc-note"></div>';
    L.DomEvent.disableClickPropagation(div);
    L.DomEvent.disableScrollPropagation(div);
    return div;
  };
  control.addTo(map);
  var select = document.getElementById("mc-select"), range = document.getElementById("mc-range"),
      note = document.getElementById("mc-note"), heading = document.getElementById("period-note");
  frames.forEach(function (f, i) { var o = document.createElement("option"); o.value = i; o.textContent = f.label; select.appendChild(o); });
  function show(i) {
    i = Math.max(0, Math.min(last, i));
    overlay.setUrl(frames[i].uri);
    select.value = i;
    if (i >= 1) { range.value = i; }
    var text = i === 0
      ? "Average of the " + last + " monthly predictions, each season weighted equally."
      : "Model estimate for " + frames[i].label + ", from that month's satellite and weather data.";
    note.textContent = text;
    if (heading) { heading.innerHTML = "Showing: <b>" + frames[i].label + "</b>"; }
  }
  select.addEventListener("change", function () { show(parseInt(select.value, 10)); });
  range.addEventListener("input", function () { show(parseInt(range.value, 10)); });
  document.getElementById("mc-prev").addEventListener("click", function () { var i = parseInt(select.value, 10); show(i <= 1 ? 1 : i - 1); });
  document.getElementById("mc-next").addEventListener("click", function () { var i = parseInt(select.value, 10); show(i < 1 ? 1 : i + 1); });
  show(0);
});
"""


def build_map(pm25, far_from_station, extent, stations, zones, path, desert_km, vmin, vmax, title, subtitle, repo_url=None, home_url=None, period_label=None, monthly=None):
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
        home_url (str | None): Link back to the summary page, shown in the box if given.
        period_label (str | None): The period the predictions describe, e.g. "March 2025 to June 2026". If given,
            it is shown in a highlighted note in the title box, in the layer name and in the colour legend.
        monthly (dict | None): Month label (for example "March 2025") to a 2-D north-up raster of that month's
            predicted PM2.5, in chronological order, same shape as `pm25`. If given, the map gets a month selector
            (drop-down, slider and previous/next buttons) that swaps the PM2.5 image, with "All months (average)"
            (the `pm25` raster) as the first and default choice; `vmin` and `vmax` then apply to every frame so the
            colours are comparable between months.

    Raises:
        ValueError: If the rasters differ in shape or `vmax` is not above `vmin`.
    """
    pm25 = np.asarray(pm25, dtype=float)
    far = np.asarray(far_from_station, dtype=bool)
    if pm25.shape != far.shape:
        raise ValueError("pm25 and far_from_station must have the same shape.")
    if not vmax > vmin:
        raise ValueError("vmax must be above vmin.")
    if monthly is not None and any(np.asarray(r).shape != pm25.shape for r in monthly.values()):
        raise ValueError("Every monthly raster must have the same shape as pm25.")
    west, east, south, north = extent

    in_period = f" in {period_label}" if period_label else ""
    if monthly:
        pm25_layer_name = "Predicted PM2.5, selected month (µg/m³)"
        legend_caption = "Predicted PM2.5 (µg/m³), the same scale for every month"
    else:
        pm25_layer_name = f"Predicted PM2.5, mean {period_label} (µg/m³)" if period_label else "Predicted PM2.5 (mean, µg/m³)"
        legend_caption = f"Mean predicted PM2.5, {period_label} (µg/m³)" if period_label else "Mean predicted PM2.5 (µg/m³)"

    pm25_mercator = to_web_mercator_rows(pm25, north, south)
    far_mercator = to_web_mercator_rows(far, north, south)
    cmap = sequential_cmap()

    def colour(raster_mercator):
        rgba = cmap(np.clip((raster_mercator - vmin) / (vmax - vmin), 0, 1))
        rgba[np.isnan(raster_mercator)] = (0, 0, 0, 0)
        return rgba

    rgba = colour(pm25_mercator)
    veil = np.zeros(far_mercator.shape + (4,))
    veil[far_mercator & ~np.isnan(pm25_mercator)] = to_rgba("#0b0b0b", 0.30)

    centre = [(south + north) / 2, (west + east) / 2]
    fmap = folium.Map(location=centre, zoom_start=5, min_zoom=4, max_zoom=12, tiles=None, control_scale=True)
    # Base maps that need no API key (CARTO's tiles now answer "API key required"). Place names are drawn
    # in their own pane above the image layers, so they stay readable over the PM2.5 colours.
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, HERE, Garmin, FAO, NOAA, USGS", name="Light grey base map (Esri)", max_native_zoom=16, control=True,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png", attr="&copy; OpenStreetMap contributors", name="OpenStreetMap",
        max_native_zoom=19, show=False, control=True,
    ).add_to(fmap)
    folium.map.CustomPane("place_names", z_index=650, pointer_events=False).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        attr="Labels &copy; Esri", name="Place names", overlay=True, control=True, max_native_zoom=16, pane="place_names",
    ).add_to(fmap)
    bounds = [[south, west], [north, east]]
    pm25_overlay = folium.raster_layers.ImageOverlay(
        image=_png_data_uri(rgba, colours=128 if monthly else None), bounds=bounds, opacity=0.88, name=pm25_layer_name, interactive=False, zindex=2,
    )
    pm25_overlay.add_to(fmap)
    folium.raster_layers.ImageOverlay(
        image=_png_data_uri(veil), bounds=bounds, opacity=1.0, name=f"Shaded: more than {desert_km:g} km from a station",
        interactive=False, zindex=3, show=True,
    ).add_to(fmap)

    station_layer = folium.FeatureGroup(name=f"Monitoring stations ({len(stations)})", show=True)
    for station in stations.itertuples():
        folium.CircleMarker(
            [station.latitude, station.longitude], radius=3.5, color="#fcfcfb", weight=1, fill=True, fill_color=ORANGE, fill_opacity=1.0,
            popup=folium.Popup(f"<b>Station {station.station_id}</b><br>{station.n_months} months of PM2.5{in_period}<br>mean observed {station.mean_pm25:.0f} µg/m³", max_width=220),
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

    legend = bcm.LinearColormap(MAP_RAMP, vmin=vmin, vmax=vmax, caption=legend_caption)
    legend.add_to(fmap)
    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)

    links = []
    if home_url:
        links.append(f'<a href="{home_url}" target="_top">&larr; Summary of the findings</a>')
    if repo_url:
        links.append(f'<a href="{repo_url}" target="_blank" rel="noopener">Code and write-up</a>')
    link = "<br>" + " &middot; ".join(links) if links else ""
    if monthly:
        note = (
            '<div style="margin-top: 8px; padding: 6px 9px; background: #fff3ec; border: 1px solid #eb6834; border-radius: 4px; font-size: 12px;">'
            f'<span id="period-note"></span><br>Choose a month from {list(monthly)[0]} to {list(monthly)[-1]} with the selector at the bottom left.</div>'
        )
    else:
        note = (
            '<div style="margin-top: 8px; padding: 6px 9px; background: #fff3ec; border: 1px solid #eb6834; border-radius: 4px; font-size: 12px;">'
            f"Period shown: <b>{period_label}</b>. An average of the monthly predictions, with each season weighted equally.</div>"
            if period_label else ""
        )
    box = f"""
    <div style="position: fixed; top: 10px; left: 55px; z-index: 9999; max-width: 340px; padding: 10px 14px;
                background: rgba(252,252,251,0.94); border-radius: 6px; border: 1px solid #e1e0d9;
                font: 13px/1.4 system-ui, -apple-system, 'Segoe UI', sans-serif; color: #0b0b0b;">
      <div style="font-weight: 700; font-size: 15px;">{title}</div>
      <div style="color: #52514e; margin-top: 3px;">{subtitle}</div>{note}{link}
    </div>"""
    fmap.get_root().html.add_child(folium.Element(box))
    fmap.get_root().header.add_child(folium.Element(f"<title>{title}</title>"))
    if monthly:
        frames = [{"label": "All months (average)", "uri": _png_data_uri(rgba, colours=128)}]
        for label, raster in monthly.items():
            frames.append({"label": label, "uri": _png_data_uri(colour(to_web_mercator_rows(np.asarray(raster, dtype=float), north, south)), colours=128)})
        fmap.get_root().header.add_child(folium.Element(MONTH_CONTROL_CSS))
        script = (MONTH_CONTROL_JS.replace("__MAP__", fmap.get_name()).replace("__OVERLAY__", pm25_overlay.get_name())
                  .replace("__FRAMES__", json.dumps(frames)))
        fmap.get_root().script.add_child(folium.Element(script))
    fmap.save(str(path))
