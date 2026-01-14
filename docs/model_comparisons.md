# 🔬 Visual Comparison of Cellular Automata Models

This document provides visual outputs of various 1D and 2D cellular automata supported in the project.

---

## Rule 30 (1D)

- Behavior: Chaotic, random-looking pattern from a simple rule
- Wolfram Class: III (chaotic)

![Rule 30](assets/rule_30.gif)

---

## Rule 90 (1D)

- Behavior: Deterministic fractals (Sierpiński triangle)
- Wolfram Class: II (periodic)

![Rule 90](assets/rule_90.gif)

---

## Rule 110 (1D)

- Behavior: Complex structures, Turing complete
- Wolfram Class: IV (edge of chaos)

![Rule 110](assets/rule_110.gif)

---

## Game of Life (2D)

- Behavior: Gliders, oscillators, chaotic growth
- Wolfram Class: IV

![Game of Life](assets/game_of_life.gif)

---

## Reaction-Diffusion (Gray-Scott)

- Behavior: Turing-like patterns from chemical interactions
- Used for: Modeling skin, coats, morphogenesis

![Reaction-Diffusion](assets/reaction_diffusion.gif)

---

## How These Were Generated

Run each model with `runner.py` and `--save-video`:

```bash
python runner.py --model rule_30 --steps 100 --dim 1 --save-video docs/assets/rule_30.gif
python runner.py --model game_of_life --steps 200 --save-video docs/assets/game_of_life.gif
```
