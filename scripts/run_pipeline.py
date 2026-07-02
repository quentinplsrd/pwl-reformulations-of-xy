# -*- coding: utf-8 -*-
import os
import sys
from datetime import timedelta
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_handling import import_QPLIB_instance_table, down_select_valid_instances
from src.experiments import build_table_of_experiments, solve_experiments_in_parallel
from src.visualization import (
    load_data,
    plot_efficiency_grid,
    plot_scalability,
    plot_dolan_more,
    plot_applicability,
    plot_approx_gap,
    generate_qplib_latex_table
)

def main():
    # Set execution context to root for file I/O
    os.chdir(project_root)
    
    # Define results directory and ensure it exists
    results_dir = os.path.join(project_root, 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Define exact path for the CSV
    result_filepath = os.path.join(results_dir, 'Exp_results_all.csv')
    
    if not os.path.isfile(result_filepath):
        # Fetch and filter external data
        df_QPLIB_instances = import_QPLIB_instance_table()
        df_QPLIB_instances = down_select_valid_instances(df_QPLIB_instances)
        
        # 2. Build the experiment table
        df_experiments = build_table_of_experiments(
            qplib_instances=df_QPLIB_instances.index,
            milp_solvers=["Gurobi", "SCIP", "HiGHS"],
            qp_solvers=["Gurobi", "SCIP"]
        )
    
        # 3. Configure Run Limits
        df_experiments['Gap tolerance'] = 0.01
        df_experiments['Time limit'] = timedelta(seconds=600)
        df_experiments['Threads'] = 1
        
        # 4. Execute Workloads
        df_exp_results = solve_experiments_in_parallel(df_experiments, max_workers=16)
        
        # Save results directly to the results folder
        df_exp_results.to_csv(result_filepath)
    
    # 5. Load the Result Set from the results folder
    print("Loading data...")
    dataset = load_data(result_filepath)
    
    # 6. Dispatch Visualizations (routing outputs to results_dir)
    print("Generating Efficiency Grid...")
    plot_efficiency_grid(dataset, output_dir=results_dir)
    
    print("Generating Scalability Plot...")
    plot_scalability(dataset, output_dir=results_dir)
    
    print("Generating Dolan-More Profile...")
    plot_dolan_more(dataset, output_dir=results_dir)
    
    print("Generating Applicability Plot...")
    plot_applicability(dataset, output_dir=results_dir)
    
    print("Generating Approx Gap Plot...")
    plot_approx_gap(dataset, output_dir=results_dir)
    
    print("\nGenerating LaTeX Table...")
    generate_qplib_latex_table(dataset)
    
    print(f"\nAll visualizations completed successfully. Check the {results_dir} folder.")

if __name__ == "__main__":
    main()