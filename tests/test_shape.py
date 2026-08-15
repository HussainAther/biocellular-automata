from src.ca_core import CellularAutomaton
from ca_rules.rule_90 import rule_90
from src.models.game_of_life import game_of_life_rule


def test_shape_1d_history():
    ca = CellularAutomaton(grid_size=101, rule_fn=rule_90, dim=1, seed=0, init="center_dot")
    history = ca.run(steps=25)
    assert history.shape == (26, 101)  # includes initial frame


def test_shape_2d_history():
    ca = CellularAutomaton(grid_size=(40, 50), rule_fn=game_of_life_rule, dim=2, seed=0, init="random")
    history = ca.run(steps=10)
    assert history.shape == (11, 40, 50)

