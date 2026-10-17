from pathlib import Path

import pytest
import yaml

from monitoring_deserts.viz import landing

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/someone/monitoring-deserts"


@pytest.fixture(scope="module")
def key():
    with open(ROOT / "reports" / "key_numbers.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def html(key, tmp_path):
    path = tmp_path / "index.html"
    landing.build_landing_page(key, path, REPO)
    return path.read_text(encoding="utf-8")


def test_the_page_quotes_the_headline_numbers(key, html):
    numbers = landing.headline_numbers(key)
    assert numbers["r2_random"] == "0.60" and numbers["r2_spatial"] == "0.27"
    assert numbers["people_50"] == "56%" and numbers["people_100"] == "27%"
    for text in (numbers["people_50"], numbers["r2_spatial"], numbers["poorest_25"], numbers["median_km"]):
        assert text in html


def test_the_period_of_the_predictions_is_stated_and_not_presented_as_live(key, html):
    assert key["data"]["period"] in html and "not a live reading" in html


def test_every_placeholder_is_filled(html):
    assert "$" not in html and "YOUR_GITHUB_USER" not in html


def test_every_figure_the_page_uses_is_served_from_docs(html):
    for name in landing.FIGURES:
        assert f'src="figures/{name}"' in html
        assert (ROOT / "docs" / "figures" / name).exists(), f"docs/figures/{name} is missing; run scripts/make_report_assets.py"


def test_links_point_at_the_given_repository_and_the_map(html):
    assert f'{REPO}/blob/main/reports/writeup.md' in html
    assert 'href="map.html"' in html and 'src="map.html"' in html
    assert "https://someone.github.io/monitoring-deserts/figures/hero_map.png" in html  # social-preview image


def test_a_repository_url_that_is_not_github_is_rejected(key, tmp_path):
    with pytest.raises(ValueError):
        landing.build_landing_page(key, tmp_path / "x.html", "https://example.org/a/b")


def test_a_missing_number_is_reported(key, tmp_path):
    broken = {k: v for k, v in key.items() if k != "paired_forest_vs_baselines"}
    with pytest.raises(KeyError):
        landing.build_landing_page(broken, tmp_path / "x.html", REPO)
