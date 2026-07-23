# Deep Reinforcement Learning-Guided Genetic Algorithms for the Flow Shop Problem

**A Centralized Deep Q-Network controller that steers a Genetic Algorithm's operators in real time — turning a static metaheuristic into an adaptive one.**

> This repository is the fifth and final phase of a broader study benchmarking optimisation paradigms on the Permutation Flow Shop Problem (PFSP): exact methods, heuristics, and metaheuristics (trajectory- and population-based) were covered in a companion repository. This phase asks a different question: **can a reinforcement learning agent learn to drive a metaheuristic better than a human-tuned, fixed configuration can?**

---

## Why This Is Interesting

Every Genetic Algorithm has to answer the same design question before it ever runs: *which selection, crossover, and mutation operators should I use, and at what rates?* The standard answer is to pick one configuration and freeze it for the entire run. But the ideal operator mix isn't static — a population early in the search benefits from aggressive exploration, while a population that is close to converging needs careful exploitation instead. A fixed GA cannot tell the difference between these two moments and cannot adapt.

This project replaces that fixed choice with a **Dueling Double Deep Q-Network (DQN)** that watches the population's health at every generation — its diversity, its progress, how long it has been stagnating, its gap to the known lower bound — and dynamically selects one of four complete operator configurations ("macro-actions") accordingly:

| Action | Profile | Selection | Crossover | Mutation | $P_c$ | $P_m$ |
|---|---|---|---|---|---|---|
| 0 | **Exploit** | Tournament | OX | Swap | 0.90 | 0.02 |
| 1 | **Balanced** | Ranking | Two-Point | Insert | 0.75 | 0.08 |
| 2 | **Explore** | Roulette | OX | Inversion | 0.60 | 0.25 |
| 3 | **Soft Explore** | Ranking | Two-Point | Scramble | 0.65 | 0.12 |

Instead of tuning individual knobs in isolation, the controller learns **when** to be aggressive and **when** to slow down — closing the loop between the optimiser and the state of its own search.

---

## How It Works

```
Population State (4 features)
   diversity · progress · stagnation · gap-to-lower-bound
                        │
                        ▼
        Linear(4→32) → ReLU → Linear(32→32) → ReLU
                        │
                  Dueling heads
             ┌────────────────────┐
          Value V(s)      Advantage A(s,a)
             └─────────┬──────────┘
                    Q(s, a)
                        │
                        ▼
           ε-greedy macro-action selection
                        │
                        ▼
     GA generation runs with the selected operator set
                        │
                        ▼
      Reward computed → transition stored → Double DQN update
```

- **Dueling architecture** separates "how good is this state" from "how much better is each action," stabilising learning when many actions have similar value.
- **Double DQN** decouples action selection from action evaluation to curb Q-value overestimation, with a target network synced every 50 gradient steps.
- **Reward shaping** gives a strong signal on genuine makespan improvement, a bonus for breaking a stagnation streak, and a mild penalty for prolonged stagnation — so the agent learns to associate the right macro-action with real progress, not noise.
- **Warm-start action masking** biases the controller toward exploitation-oriented actions early in a run and unlocks the highest-perturbation action only once stagnation is detected, preventing premature over-exploration while still letting every transition contribute to training.
- **Surrogate-assisted evaluation:** for very large instances, a **LightGBM** regression model (trained offline on engineered population/schedule features) pre-screens offspring so that only the most promising candidates receive an exact, full makespan evaluation — keeping large-instance runs computationally tractable.
- **Persistence by design:** network weights, the replay buffer, and epsilon are checkpointed to disk after every instance, so the controller keeps learning across sessions rather than starting from scratch each run.

The DQN-controlled GA is benchmarked head-to-head against a **fixed-operator reference GA** (roulette selection, two-point crossover, scramble mutation) on identical population sizes and generation budgets, and both are scored against the certified **Taillard upper bounds** using:

$$\text{Gap}\% = \frac{C_{max} - UB}{UB} \times 100$$

---

## Repository Structure

```
.
├── centralized_dqn_ga_optisolve_v13_2.ipynb   # Main notebook — full pipeline & experiments
├── checkpoints/                                # Saved DQN weights, target network, replay buffer, epsilon
├── surrogate_models/                           # Pre-trained LightGBM makespan surrogates (per size class)
└── tai/                                        # Taillard (.fsp) benchmark instances
```

## Notebook Walkthrough

| Section | Content |
|---|---|
| 0 | Imports, global configuration, size-dependent GA parameters, DQN hyperparameters |
| 1 | Taillard instance loader |
| 2 | Core makespan evaluation (DP recurrence + single-pass partial-checkpoint extraction) and NEH seeding |
| 3 | Hybrid population initialization (NEH + perturbations + random) |
| 4 | Genetic operators — crossover (OX, Two-Point, PMX), mutation (Swap, Insert, Inversion, Scramble), selection (Tournament, Ranking, Roulette), elitist replacement |
| 5 | LightGBM surrogate model for large-instance offspring pre-screening |
| 6 | Centralized DQN controller — network design, state representation, Double DQN training, warm-start masking, reward function |
| 7 | DQN-controlled GA main loop |
| 8 | Fixed-operator reference GA (baseline) |
| 9 | Controller initialization / checkpoint restoration |
| 10 | Notes on the online-learning schedule (no offline pre-training — the agent learns live during benchmarking) |
| 11 | Benchmark execution across all available Taillard instances |
| 12 | Results — ARPD gap comparison between DQN-GA and the baseline |
| 12b | Controller diagnostics — loss curve, action-distribution breakdown by instance size, and an honest discussion of current limitations |

## Requirements

```
numpy
pandas
torch
lightgbm
joblib
```

```bash
pip install numpy pandas torch lightgbm joblib
```

Place Taillard `.fsp` benchmark files under `tai/` before running the notebook. Trained surrogate models are expected under `surrogate_models/`; if none are found, the notebook falls back to exact evaluation for every instance.

## Honest Status

This is an active research notebook, not a finished product — Section 12b documents the controller's current diagnostics candidly, including cases where the learned policy still favors exploration more heavily than expected on some instance sizes. The version history embedded in the notebook (v5 → v13) tracks a sequence of concrete, diagnosed fixes (reward scaling, target-network update frequency, per-size epsilon floors, stratified replay buffering, budgeted early stopping) rather than presenting a black-box result — the goal is to make the reinforcement-learning loop itself inspectable and improvable.

---

*Part of a five-phase study of optimisation methods for the Flow Shop Scheduling Problem: exact algorithms → heuristics → trajectory metaheuristics → population-based metaheuristics → this project, reinforcement-learning-guided metaheuristics.*
