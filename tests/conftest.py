import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def station_months():
    """60 stations in three clusters, each with three monthly rows (made-up data)."""
    rng = np.random.default_rng(0)
    centres = np.array([[72.0, 12.0], [78.0, 22.0], [85.0, 27.0]])
    rows = []
    for i in range(60):
        lon, lat = centres[i % 3] + rng.normal(0, 0.8, 2)
        for month in pd.period_range("2025-03", periods=3, freq="M"):
            rows.append({"station_id": f"s{i}", "longitude": lon, "latitude": lat, "month": month})
    df = pd.DataFrame(rows)
    df["x1"] = rng.normal(size=len(df))
    df["x2"] = rng.normal(size=len(df))
    df["y"] = 2 * df["x1"] + rng.normal(0, 0.3, len(df))
    return df
