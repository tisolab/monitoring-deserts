import numpy as np
import pytest
from scipy.stats import spearmanr

from monitoring_deserts.features import equity


def test_weighted_mean():
    assert equity.weighted_mean([1, 3], [1, 3]) == 2.5
    assert np.isnan(equity.weighted_mean([1, 2], [0, 0]))
    with pytest.raises(ValueError):
        equity.weighted_mean([1], [-1])


def test_weighted_quantile_groups_hold_equal_weight_and_keep_ties_together():
    rng = np.random.default_rng(0)
    values, weights = rng.normal(size=10000), rng.exponential(1, 10000)
    labels, cutoffs = equity.weighted_quantile_groups(values, weights, 5)
    shares = [weights[labels == k].sum() / weights.sum() for k in range(5)]
    assert shares == pytest.approx([0.2] * 5, abs=0.01)
    assert labels[np.argmin(values)] == 0 and labels[np.argmax(values)] == 4 and len(cutoffs) == 4
    tied, _ = equity.weighted_quantile_groups(np.r_[np.zeros(50), np.ones(50)], np.ones(100), 2)
    assert set(tied[:50]) == {0.0} and set(tied[50:]) == {1.0}


def test_weighted_quantile_groups_leave_unusable_rows_ungrouped():
    values = np.arange(20.0)
    values[:2] = np.nan
    weights = np.ones(20)
    weights[2:4] = 0
    labels, _ = equity.weighted_quantile_groups(values, weights, 4)
    assert np.isnan(labels[:4]).all() and not np.isnan(labels[4:]).any()
    with pytest.raises(ValueError):
        equity.weighted_quantile_groups([1, 2], [1, 2], 5)


def test_group_status_table_shares_and_weighted_means():
    groups = np.array([0, 0, 1, 1, np.nan])
    in_desert = np.array([True, False, True, True, True])
    population = np.array([10, 30, 20, 20, 5.0])
    values = np.array([10, 20, 30, 50, 99.0])
    table = equity.group_status_table(groups, in_desert, population, values)
    assert table.loc[0, "share_in_desert"] == 0.25 and table.loc[1, "share_in_desert"] == 1.0
    assert table.loc[0, "mean_value_monitored"] == 20 and table.loc[0, "mean_value_desert"] == 10
    assert np.isnan(table.loc[1, "mean_value_monitored"]) and table.loc[1, "mean_value_desert"] == pytest.approx(40)
    with pytest.raises(ValueError):
        equity.group_status_table(groups, in_desert, np.r_[population[:4], np.nan])


def test_partial_spearman_matches_spearman_and_removes_a_common_driver():
    rng = np.random.default_rng(0)
    x = rng.normal(size=3000)
    y = x + rng.normal(size=3000)
    assert equity.partial_spearman(x, y) == pytest.approx(spearmanr(x, y)[0], abs=1e-9)
    z = rng.normal(size=3000)
    a, b = z + rng.normal(size=3000) * 0.3, z + rng.normal(size=3000) * 0.3
    assert spearmanr(a, b)[0] > 0.8
    assert abs(equity.partial_spearman(a, b, z)) < 0.1
    assert np.isnan(equity.partial_spearman(a, a, a))
    with pytest.raises(ValueError):
        equity.partial_spearman([1, 2, np.nan, 4], [1, 2, 3, 4])


def test_block_bootstrap_widens_the_interval_when_blocks_are_correlated():
    rng = np.random.default_rng(0)
    blocks = np.repeat(np.arange(40), 50)
    data = np.repeat(rng.normal(size=40), 50) + rng.normal(size=2000) * 0.1
    result = equity.block_bootstrap(lambda i: data[i].mean(), blocks, n_boot=300)
    per_cell = equity.block_bootstrap(lambda i: data[i].mean(), np.arange(2000), n_boot=300)
    assert result["lower"] < result["estimate"] < result["upper"] and result["n_blocks"] == 40
    assert result["std"] > 3 * per_cell["std"]
    same = equity.block_bootstrap(lambda i: data[i].mean(), blocks, n_boot=50, random_state=1)
    assert same == equity.block_bootstrap(lambda i: data[i].mean(), blocks, n_boot=50, random_state=1)


def test_block_bootstrap_validates_inputs():
    with pytest.raises(ValueError):
        equity.block_bootstrap(lambda i: 0.0, np.arange(3), n_boot=20)  # fewer than 5 blocks
    with pytest.raises(ValueError):
        equity.block_bootstrap(lambda i: 0.0, np.arange(100), n_boot=5)
    with pytest.raises(ValueError):
        equity.block_bootstrap(lambda i: 0.0, np.arange(100), level=1)
