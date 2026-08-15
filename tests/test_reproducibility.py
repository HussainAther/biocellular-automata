import numpy as np

from src.ca_core import CellularAutomaton
from ca_rules.rule_110 import rule_110
from ca_rules.wolfram_rule_30 import wolfram_rule_30


def test_same_seed_same_history_1d():
    ca1 = CellularAutomaton(grid_size=101, rule_fn=rule_110, dim=1, seed=123, init="random")
    ca2 = CellularAutomaton(grid_size=101, rule_fn=rule_110, dim=1, seed=123, init="random")

    h1 = ca1.run(steps=50)
    h2 = ca2.run(steps=50)

    assert np.array_equal(h1, h2), "Histories should match for same seed/init."


def test_different_seed_different_history_1d():
    ca1 = CellularAutomaton(grid_size=101, rule_fn=wolfram_rule_30, dim=1, seed=1, init="random")
    ca2 = CellularAutomaton(grid_size=101, rule_fn=wolfram_rule_30, dim=1, seed=2, init="random")

    h1 = ca1.run(steps=50)
    h2 = ca2.run(steps=50)

    # Very likely different; if ever equal, it's an extreme coincidence.
    assert not np.array_equal(h1, h2), "Histories should differ for different seeds."

