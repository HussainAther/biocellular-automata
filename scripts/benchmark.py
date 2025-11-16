import time, numpy as np
from src.ca_core import CellularAutomaton
from ca_rules.rule_110 import rule_110
for width in (257, 1025, 4097):
    ca = CellularAutomaton(width, rule_110, dim=1, seed=0)
    t0=time.perf_counter(); ca.run(steps=200); dt=time.perf_counter()-t0
    print(width, f"{dt:.3f}s")

