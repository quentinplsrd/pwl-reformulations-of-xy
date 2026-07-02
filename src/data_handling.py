# -*- coding: utf-8 -*-
import os
from io import StringIO
import requests
import certifi
import pandas as pd
import numpy as np
import pyqplib

def import_QPLIB_instance_table():
    print('Importing QPLIB instance table')
    url = "https://qplib.zib.de/instances.html"

    response = requests.get(url, verify=certifi.where(), timeout=30)
    response.raise_for_status()

    df = pd.read_html(StringIO(response.text))[0]

    df = df.iloc[:-1, :]
    df['Instance'] = df['Instance'].str.split(' ').str[0]
    df.set_index('Instance', inplace=True)
    
    return df

def down_select_valid_instances(df, max_vars=200, max_cons=200):
    print('Down-selecting the instances')
    print(f"Eliminating instances with more than {max_vars} variables or {max_cons} constraints")
    
    df = df.loc[(df['TotalVars.'] <= max_vars) & (df['Total Cons.'] <= max_cons)]
    df_valid_instances = pd.Series(index=df.index, data=True)

    # Define the specific directory for instance data
    data_dir = os.path.join(os.getcwd(), 'data', 'instances')
    os.makedirs(data_dir, exist_ok=True) # Create it if it doesn't exist

    for instance in df_valid_instances.index:
        file_name = f"QPLIB_{instance}.qplib"
        file_path = os.path.join(data_dir, file_name)
        
        # Check for the file in the data/instances/ directory
        if not os.path.isfile(file_path):
            url = f"https://qplib.zib.de/qplib/{file_name}"
            response = requests.get(url, verify=certifi.where(), timeout=30)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)
        
        try:
            # Tell pyqplib to read from the specific file path
            problem = pyqplib.read_problem(file_path)
        except:
            print(f"{file_name} could not be loaded")
            df_valid_instances[instance] = False
            continue
        
        var_types = [var.name for var in problem.var_types]
        var_lb = problem.var_lb
        var_ub = problem.var_ub
        m = problem.num_cons
        n = problem.num_vars

        H_all = [problem.obj.hess(0)]
        if hasattr(problem.constraints, 'hess_mats'):
            for i in range(m):
                H_all.append(problem.constraints.hess_mats[i].full())
        
        set_active_vars = set()
        for H in H_all:
            if len(set_active_vars) == n:
                break
            set_active_vars.update(H.row)
            set_active_vars.update(H.col)
            
        if any(var_types[v] != 'CONTINUOUS' for v in set_active_vars):
            df_valid_instances[instance] = False
            print(f"{file_name}: some variables involved in quadratic terms are not continuous")
            continue
        
        if any((np.isinf(var_lb[v]) | np.isinf(var_ub[v])) for v in set_active_vars):
            df_valid_instances[instance] = False
            print(f"{file_name}: some variables involved in quadratic terms are not bounded")
            continue
        
        found_square_term = False
        for H in H_all:
            if abs(H.diagonal()).max() > 0:
                found_square_term = True
                break
        
        if found_square_term:
            df_valid_instances[instance] = False
            print(f"{file_name}: some variables are squared")
            continue
        
        print(f"{file_name}: the instance is valid")
        
    df = df.loc[df_valid_instances]
    print(f"The down-selection resulted in {len(df)} valid instances")

    return df