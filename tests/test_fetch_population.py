import numpy as np
import pytest

from monitoring_deserts.data import fetch_population as fp

LONS = np.round(70 + 0.05 * np.arange(10), 9)
LATS = np.round(10 + 0.05 * np.arange(8), 9)


def test_locate_grid_cells_finds_the_containing_cell():
    col, lat_idx = fp.locate_grid_cells([70.0, 70.12, 70.449], [10.0, 10.16, 10.349], LONS, LATS, 0.05)
    assert col.tolist() == [0, 2, 9]
    assert lat_idx.tolist() == [0, 3, 7]


def test_a_point_on_a_shared_edge_belongs_to_the_cell_north_and_east():
    col, lat_idx = fp.locate_grid_cells([70.025], [10.025], LONS, LATS, 0.05)
    assert (col[0], lat_idx[0]) == (1, 1)


def test_locate_grid_cells_rejects_points_outside_and_bad_input():
    with pytest.raises(ValueError):
        fp.locate_grid_cells([80.0], [10.0], LONS, LATS, 0.05)
    with pytest.raises(ValueError):
        fp.locate_grid_cells([70.0, np.nan], [10.0, 10.0], LONS, LATS, 0.05)
    with pytest.raises(ValueError):
        fp.locate_grid_cells([70.0], [10.0], LONS, LATS, 0)
