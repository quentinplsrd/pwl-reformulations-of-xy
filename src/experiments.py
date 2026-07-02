# -*- coding: utf-8 -*-
import os
import sys
from contextlib import contextmanager

import pyqplib
import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from ortools.math_opt.python import mathopt
from src.models import build_NCQP_model, build_model_from_QPLIB

@contextmanager
def suppress_c_stdout_stderr():
    """Redirects C-level stdout and stderr to devnull."""
    null_fd = os.open(os.devnull, os.O_RDWR)
    save_stdout = os.dup(1)
    save_stderr = os.dup(2)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(save_stdout, 1)
        os.dup2(save_stderr, 2)
        os.close(null_fd)
        os.close(save_stdout)
        os.close(save_stderr)

def build_table_of_experiments(seeds=range(10), 
                               sequence_lengths=[24, 48, 96, 168], 
                               qplib_instances=[], 
                               milp_solvers=["Gurobi", "SCIP", "HiGHS"], 
                               qp_solvers=["Gurobi", "SCIP"], 
                               cpwl_representations=["Triangle", "Square", "DC"], 
                               degrees=[1, 2, 3, 4]):
    
    print("\nBuilding the experiment table")
    ncqp_instances = pd.DataFrame(product(seeds, sequence_lengths), columns=["Random seed", "Sequence length"])
    ncqp_instances["Instance"] = "NCQP_T_" + ncqp_instances["Sequence length"].map(lambda x: f"{x:04d}") + "_seed_" + ncqp_instances["Random seed"].map(lambda x: f"{x:03d}")
    ncqp_instances["Instance family"] = "NCQP"

    qplib = pd.DataFrame({"Instance": qplib_instances, "Instance family": "QPLIB", "Random seed": pd.NA, "Sequence length": pd.NA})

    instances = pd.concat([ncqp_instances, qplib], ignore_index=True)

    milp_cases = instances.merge(pd.DataFrame({"Solver": milp_solvers}), how="cross") \
                          .merge(pd.DataFrame({"CPWL representation": cpwl_representations}), how="cross") \
                          .merge(pd.DataFrame({"Degree of accuracy": degrees}), how="cross")
    milp_cases["Problem type"] = "MILP"

    qp_cases = instances.merge(pd.DataFrame({"Solver": qp_solvers}), how="cross")
    qp_cases["Problem type"] = "QP"
    qp_cases["CPWL representation"] = pd.NA
    qp_cases["Degree of accuracy"] = pd.NA

    table_experiments = pd.concat([milp_cases, qp_cases], ignore_index=True)
    cols = ["Instance family", "Instance", "Sequence length", "Random seed", "Problem type", "CPWL representation", "Degree of accuracy", "Solver"]
    return table_experiments[cols].sort_values(by=cols).reset_index(drop=True)

def build_experiment_model(dict_exp):
    family = dict_exp["Instance family"]
    quadratic = (dict_exp["Problem type"] == "QP")
    N = 3
    partition_method = 'Square'
    
    if not quadratic:
        N = dict_exp["Degree of accuracy"]
        partition_method = dict_exp["CPWL representation"]
    
    if family == 'NCQP':
        model, weight, X, Y = build_NCQP_model(T=dict_exp["Sequence length"], seed=dict_exp["Random seed"], 
                                               quadratic=quadratic, N=N, partition_method=partition_method)
    elif family == 'QPLIB':
        file = f"QPLIB_{dict_exp['Instance']}.qplib"
        # Assuming QPLIB instances are now routed correctly to the data/instances folder
        filepath = os.path.join(os.getcwd(), 'data', 'instances', file)
        problem = pyqplib.read_problem(filepath)
        model = build_model_from_QPLIB(problem, quadratic=quadratic, N=N, partition_method=partition_method)
        weight, X, Y = None, None, None
    else:
        raise ValueError(f'{family} is not a valid instance family')
    
    return model, weight, X, Y

def solve_experiment_model(dict_exp, enable_output=False):
    worker_pid = os.getpid()
    
    solver_map = {'Gurobi': mathopt.SolverType.GUROBI, 'HiGHS': mathopt.SolverType.HIGHS, 'SCIP': mathopt.SolverType.GSCIP}
    threads = dict_exp['Threads'] if dict_exp['Solver'] != 'HiGHS' else None
    params = mathopt.SolveParameters(enable_output=enable_output, 
                                     relative_gap_tolerance=dict_exp['Gap tolerance'], 
                                     time_limit=dict_exp['Time limit'], threads=threads)
    
    # Wrap the C++ execution block in the silencer
    with suppress_c_stdout_stderr():
        model, weight, X, Y = build_experiment_model(dict_exp)
        result = mathopt.solve(model, solver_map[dict_exp['Solver']], params=params)
    
    status = result.termination.reason.name
    
    objective_value = np.nan
    quad_objective_value = np.nan
    if result.termination.reason in [mathopt.TerminationReason.OPTIMAL, mathopt.TerminationReason.FEASIBLE]:
        objective_value = result.objective_value()
        if dict_exp['Instance family'] == 'NCQP':
            X_sol, Y_sol = np.array(result.variable_values(X)), np.array(result.variable_values(Y))
            quad_objective_value = sum(weight * X_sol * Y_sol)
            
    dual_bound = result.dual_bound()
    solve_time = result.solve_stats.solve_time.total_seconds()
    
    return dict(dict_exp) | {
        "worker_pid": worker_pid,
        "Status": status,
        "Objective value": objective_value,
        "Quadratic objective value": quad_objective_value,
        "Dual bound": dual_bound,
        "Relative gap": abs(objective_value - dual_bound) / abs(objective_value) if objective_value != 0. else np.inf,
        "Solve time": solve_time
    }

def solve_experiments_in_parallel(df, max_workers):
    dict_experiments = df.to_dict(orient="records")
    results = []
    
    print(f"\nSolving {len(df)} experiments on {max_workers} workers")
    
    case_info_widths = {
        "Instance": 20,
        "Problem type": 4,
        "CPWL representation": 8,
        "Degree of accuracy": 4,
        "Solver": 6,
    }
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(solve_experiment_model, exp): exp for exp in dict_experiments}
        n_done, n_total = 0, len(futures)
        n_charac = len(str(n_total))

        for future in as_completed(futures):
            case = futures[future]
            n_done += 1
            case_info_str = " | ".join(
                f"{str(case[k]):<{case_info_widths[k]}}"
                for k in case_info_widths.keys()
            )
            try:
                results.append(future.result())
                print(f"[{n_done:0{n_charac}d}/{n_total}] {case_info_str}", flush=True)
            except Exception as e:
                results.append({**case, "status": "FAILED", "error": str(e)})
                print(f"[{n_done}/{n_total}] {case_info_str} Failed with error: {e}", flush=True)

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)
    
    # Define the exact sorting hierarchy
    sort_columns = [
        "Instance family",
        "Instance",
        "Problem type",
        "CPWL representation",
        "Degree of accuracy",
        "Solver"
    ]
    
    # Return the sorted DataFrame with a clean index
    return df_results.sort_values(by=sort_columns).reset_index(drop=True)