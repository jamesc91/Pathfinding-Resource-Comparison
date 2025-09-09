# Grid Pathfinding Benchmarks — Dijkstra vs A* vs Q-learning

**TL;DR:** Generates a 2D grid with obstacles, then compares **Dijkstra**, **A***, and **tabular Q-learning** (train once, **inference only** for the benchmark). Prints **runtime**, **peak memory** (via `tracemalloc`), **core array sizes**, **expanded nodes**, and **path length**, and plots the resulting paths.

## What’s inside
- **`GridWorld`** environment with walls, start/goal, and step function (bump penalty, goal reward).
- **Dijkstra / A*** implementations with predecessor tracking and path reconstruction.
- **Tabular Q-learning** trainer + **greedy rollout** for evaluation (inference is what’s timed in the benchmark).
- **Benchmark harness** that measures wall-clock time and **peak memory** via `tracemalloc`, and overlays paths on a matplotlib grid.

## Algorithms
- **Dijkstra:** uniform-cost search over 4-connected grid; reports path, total cost, and nodes popped.
- **A***: Manhattan-heuristic guided search; typically fewer node pops; same reporting.
- **Q-learning (tabular):** trains a state–action value table with optional potential-based shaping; benchmark uses **greedy rollout** for inference-time comparison.

## Quick start
```bash
# Python 3.9+ recommended
pip install -r requirements.txt
python grid_pathfinding_benchmarks.py
```

### `requirements.txt`
```
numpy
matplotlib
```

## What you’ll see (sample format)
```
== Benchmark (single query; RL inference only) ==
Dijkstra:      time=... ms | peak_mem=... KB | core_arrays≈... KB | nodes=... | path_len=...
A*:            time=... ms | peak_mem=... KB | core_arrays≈... KB | nodes=... | path_len=...
Q-learn TRAIN: time=... s  | peak_mem=... KB
Q-learn INFER: time=... ms | peak_mem=... KB | core_arrays≈... KB | path_len=...
```
A plot window will open showing the grid (black squares = obstacles) with **Start**, **Goal**, and the paths found by each method overlaid.

## How it works
- **Problem generation:** `generate_problem(...)` creates a random obstacle grid, ensures start/goal are free, and retries until **A*** can find a valid path (fallback is an empty grid).
- **Metrics:** `timed_peakmem(fn, ...)` runs each algorithm and records wall-clock time and **peak bytes** with `tracemalloc`. Node expansions (“popped”) come from the priority queue loops.
- **Core arrays (approx):** For planners, calculated from distances & predecessor arrays; for RL inference, the `Q` table size.

## Tuning & options
Edit defaults in the source if you want to experiment:
- **Grid size & density:** `generate_problem(nrows=15, ncols=15, obstacle_prob=0.22, seed=42)`
- **Q-learning:** `episodes, alpha, gamma, eps_*`, `max_steps`, and `shaping=True/False`
- **Rollout:** `rl_rollout(..., max_steps=2000)`
All knobs are top-level function arguments for easy tweaking.

## Notes & limitations
- The Q-learning agent is **tabular** and trained on a single static grid; the benchmark reports **inference** time to make comparison fair to planners.
- Manhattan distance is an **admissible** heuristic for 4-connected grids with unit edge costs.
- Plotting uses `matplotlib`; disable plotting if you’re running headless.

## Roadmap (nice-to-haves)
- CLI flags (argparse) for grid size, obstacle rate, seeds.
- Multi-trial averages with confidence intervals.
- 8-connected moves, weighted costs, or diagonal heuristics.
- Q-learning generalization across multiple random grids.

## Repo “Topics”
`pathfinding` · `dijkstra` · `a-star` · `reinforcement-learning` · `q-learning` · `gridworld` · `benchmark` · `tracemalloc` · `python` · `numpy` · `matplotlib`
