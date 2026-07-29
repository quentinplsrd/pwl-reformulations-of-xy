# Piecewise Linear Approximations for Bilinear Reformulations

This repository contains a functional, modular computational pipeline for evaluating and comparing piecewise linear (PWL) approximations in bilinear reformulations (such as those found in NCQP and QPLIB instances). 

The pipeline handles automated data acquisition, model generation, parallelized solver execution (via OR-Tools for Gurobi, SCIP, and HiGHS), and artifact visualization.

## Repository Structure

```text
pwl-reformulations/
├── src/
│   ├── data_handling.py       # External data fetching (QPLIB)
│   ├── geometry.py            # Mathematical/computational geometry functions
│   ├── models.py              # MILP and QP model construction using math_opt
│   ├── experiments.py         # Parallel execution and solver configuration
│   └── visualization.py       # Matplotlib plotting and LaTeX table generation
├── scripts/
│   └── run_pipeline.py        # Top-level orchestration script
├── data/                      
│   ├── instances/             # Downloaded .qplib instances (git-ignored)
│   └── results/               # Generated CSVs and figures (git-ignored)
├── pyproject.toml             # Project metadata and dependencies
└── README.md
```

## Setup and Installation

This project uses `uv` for fast dependency management.

1. Install `uv` by following the instructions at [docs.astral.sh/uv/getting-started/installation](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone this repository and navigate to the project root:
   ```bash
   git clone https://github.com/quentinplsrd/pwl-reformulations-of-xy
   cd pwl-reformulations-of-xy
   ```
3. Create/update the projects environment with:
   ```bash
   uv sync
   ```

## Execution

The entire pipeline is orchestrated through a single script. Execute it from the root of the repository:

```bash
uv run scripts/run_pipeline.py
```

**Workflow sequence:**
1. Validates and downloads necessary QPLIB instances to `data/instances/`.
2. Builds the experiment matrix.
3. Executes models in parallel across specified solvers (Gurobi, SCIP, HiGHS).
4. Saves raw metrics to `data/results/Exp_results_all.csv`.
5. Generates performance profiles, efficiency grids, and approximation gap boxplots in `data/results/`.


If Gurobi isn't installed properly (or installed at all), you will see the following message and the Gurobi solutions will be skipped.
> Gurobi solver not detected - excluding from experiments.
