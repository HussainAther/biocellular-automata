# runner.py

from src.ca_core import CellularAutomaton
from src.models.game_of_life import game_of_life_rule
from utils.visualize import animate_history

def main():
    ca = CellularAutomaton(
        grid_size=(50, 50),
        rule_fn=game_of_life_rule,
        num_states=2,
        neighborhood='Moore',
        dim=2
    )

    history = ca.run(steps=100)
    animate_history(history, interval=100)

if __name__ == "__main__":
    main()

