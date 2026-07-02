# -*- coding: utf-8 -*-
import random
import numpy as np
from ortools.math_opt.python import mathopt
from src.geometry import (
    list_faces_from_N,
    equations_sum_convex,
    equations_from_faces_3d
)

def binary_rep(k, N):
    p = (N - 1).bit_length()
    return ((k >> np.arange(p)) & 1).astype(np.int32)

def add_pwl_constraints(model, x, y, z, COEFFS, EQUATIONS, logarithmic_encoding=False, name=''):
    N_C = len(COEFFS)
    z_c = [model.add_variable(name=f"z_c({name},{cc})") for cc in range(N_C)]
    model.add_linear_constraint(z == sum(z_c), name=f"sum_convex({name})")  
    
    xx, yy, zz, bv, gv = [], [], [], [], []
    
    for cc in range(N_C):
        coeffs = list(COEFFS[cc])
        equations = list(EQUATIONS[cc])
        NN = len(equations)
        P = (NN - 1).bit_length()
        
        xx_c = [model.add_variable(name=f"xx({name},{cc},{i})", lb=0., ub=1.) for i in range(NN)]
        yy_c = [model.add_variable(name=f"yy({name},{cc},{i})", lb=0., ub=1.) for i in range(NN)]
        zz_c = [model.add_variable(name=f"zz({name},{cc},{i})") for i in range(NN)]
        
        if logarithmic_encoding:
            bv_c = [model.add_variable(name=f"bv({name},{cc},{i})", lb=0., ub=1.) for i in range(NN)]
            gv_c = [model.add_binary_variable(name=f"gv({name},{cc},{q})") for q in range(P)]
        else:
            bv_c = [model.add_binary_variable(name=f"bv({name},{cc},{i})") for i in range(NN)]
    
        xx.append(xx_c)
        yy.append(yy_c)
        zz.append(zz_c)
        bv.append(bv_c)
        if logarithmic_encoding:
            gv.append(gv_c)
        
        model.add_linear_constraint(x == sum(xx_c), name=f"sum_xx({name},{cc})")
        model.add_linear_constraint(y == sum(yy_c), name=f"sum_yy({name},{cc})")
        model.add_linear_constraint(z_c[cc] == sum(zz_c), name=f"sum_zz({name},{cc})")
        model.add_linear_constraint(sum(bv_c) == 1., name=f"sum_bv({name},{cc})")
        
        if logarithmic_encoding:
            binary_array = np.array([binary_rep(i, NN) for i in range(NN)])
            for q in range(P):
                model.add_linear_constraint(
                    sum([bv_c[i] for i in range(NN) if binary_array[i, q] == 1]) <= gv_c[q],
                    name=f"gv_lb({name},{cc},{q})")
                model.add_linear_constraint(
                    sum([bv_c[i] for i in range(NN) if binary_array[i, q] == 0]) <= 1 - gv_c[q],
                    name=f"gv_ub({name},{cc},{q})")

        for i in range(NN):
            a, b, c = coeffs[i]
            model.add_linear_constraint(zz_c[i] == a * xx_c[i] + b * yy_c[i] + c * bv_c[i])
            for j in range(len(equations[i])):
                aa, bb, cc_eq = equations[i][j]
                model.add_linear_constraint(aa * xx_c[i] + bb * yy_c[i] + cc_eq * bv_c[i] <= 0.)
        
        if N_C == 2:
            for i in range(NN):
                model.add_linear_constraint(xx_c[i] <= bv_c[i])
                model.add_linear_constraint(yy_c[i] <= bv_c[i])
        
    return {'x': x, 'y': y, 'z': z, 'xx': xx, 'yy': yy, 'zz': zz, 'z_c': z_c, 'bv': bv, 'gv': gv}

def build_NCQP_model(T=48, seed=0, quadratic=True, N=3, partition_method='Square', logarithmic_encoding=False):
    random.seed(seed)
    
    S_initial = 100.0
    S_max = 250.0
    X_max = 40.0
    Y_max = 5.0 + 0.2 * S_max
    Y_min = 5.0
    
    inflow = np.array([random.uniform(10.0, 30.0) for _ in range(T)])
    weight = np.array([random.uniform(50.0, 150.0) for _ in range(T)])
    
    model = mathopt.Model(name="NCQP")
    
    S = [model.add_variable(lb=0.0, ub=S_max, name=f"Storage({t})") for t in range(T)]
    X = [model.add_variable(lb=0.0, ub=X_max, name=f"Allocation({t})") for t in range(T)]
    Y = [model.add_variable(lb=Y_min, ub=Y_max, name=f"Multiplier({t})") for t in range(T)]

    for t in range(T):
        if t == 0:
            model.add_linear_constraint(S[t] == S_initial + inflow[t] - X[t])
        else:
            model.add_linear_constraint(S[t] == S[t-1] + inflow[t] - X[t])
        model.add_linear_constraint(Y[t] == 5.0 + 0.2 * S[t])

    if quadratic:
        objective_expr = sum(weight[t] * X[t] * Y[t] for t in range(T))
    else:
        Z = [model.add_variable(name=f"Z_{t}") for t in range(T)]
        x = [model.add_variable(lb=0., ub=1., name=f"x_{t}") for t in range(T)]
        y = [model.add_variable(lb=0., ub=1., name=f"y_{t}") for t in range(T)]
        z = [model.add_variable(lb=0., ub=1., name=f"z_{t}") for t in range(T)]
        
        if partition_method not in ['Triangle', 'Square', 'DC']:
            print(f'The representation "{partition_method}" is not valid.')
            return
        
        if partition_method in ['Triangle', 'Square']:
            faces = list_faces_from_N(N, method=partition_method)
            list_coeffs, list_equations = equations_from_faces_3d(faces)
            COEFFS, EQUATIONS = [list_coeffs], [list_equations]
        else:
            list_coeffs_j, list_equations_j, list_coeffs_k, list_equations_k = equations_sum_convex(N)
            COEFFS = [list_coeffs_j, list_coeffs_k]
            EQUATIONS = [list_equations_j, list_equations_k]

        for t in range(T):
            model.add_linear_constraint(X[t] == 0.0 + (X_max - 0.0) * x[t], name=f"X_to_x({t})")
            model.add_linear_constraint(Y[t] == Y_min + (Y_max - Y_min) * y[t], name=f"Y_to_y({t})")
            model.add_linear_constraint(Z[t] == 0.0 * Y_min + Y_min * (X_max - 0.0) * x[t] + 
                                        0.0 * (Y_max - Y_min) * y[t] + (Y_max - Y_min) * (X_max - 0.0) * z[t], 
                                        name=f"Z_to_z({t})")
        
            add_pwl_constraints(model, x[t], y[t], z[t], COEFFS, EQUATIONS, 
                                logarithmic_encoding=logarithmic_encoding, name=f'{t}')

        objective_expr = sum(weight[t] * Z[t] for t in range(T))

    model.maximize(objective_expr)
    return model, weight, X, Y

def build_model_from_QPLIB(problem, quadratic=True, N=3, partition_method='Square', logarithmic_encoding=False):
    n = problem.num_vars
    m = problem.num_cons
    var_types = [var.name for var in problem.var_types]
    var_lb = problem.var_lb
    var_ub = problem.var_ub
    cons_lb = problem.constraints.lb
    cons_ub = problem.constraints.ub
    cons_lin = problem.constraints.mat.tocsr()
    
    model = mathopt.Model(name='QPLIB')
        
    variables = []
    for i in range(n):
        if var_types[i] == 'CONTINUOUS':
            variable = model.add_variable(lb=var_lb[i], ub=var_ub[i], name=f"variable({i})")
        elif var_types[i] == 'BINARY':
            variable = model.add_binary_variable(name=f"variable({i})")
        elif var_types[i] == 'INTEGER':
            variable = model.add_integer_variable(lb=var_lb[i], ub=var_ub[i], name=f"variable({i})")
        variables.append(variable)
        
    obj_sense = problem.obj.sense.name
    obj_lin = problem.obj.lin
    obj_hess = problem.obj.hess(0)
    
    saddle_list = {'objective': [], 'constraints': []}
    for i, j, a in zip(obj_hess.row, obj_hess.col, obj_hess.data):
        if j < i: continue
        saddle_list['objective'].append([i, j, a])
        
    if hasattr(problem.constraints, 'hess_mats'):
        H_cons_list = [problem.constraints.hess_mats[k].full() for k in range(m)]
        for k in range(m):
            H_cons = H_cons_list[k]
            sub_list = []
            for i, j, a in zip(H_cons.row, H_cons.col, H_cons.data):
                if j < i: continue
                sub_list.append([i, j, a])
            saddle_list['constraints'].append(sub_list)
            
    unique_saddles = np.unique(np.array(saddle_list['objective'] + 
                                        [item for sub_list in saddle_list['constraints'] for item in sub_list])[:, :2].astype(int), axis=0)
    dict_unique_saddles = {(i, j): k for k, (i, j) in enumerate(unique_saddles)}
    variables_in_saddle = np.unique(unique_saddles)
    variables_in_saddle_bool = np.full(n, False)
    variables_in_saddle_bool[variables_in_saddle] = True
    
    if not quadratic:
        Z = [model.add_variable(name=f"Z({i,j})") for i, j in unique_saddles]
        x = [model.add_variable(lb=0., ub=1., name=f"x({i})") if variables_in_saddle_bool[i] else None for i in range(n)]
        z = [model.add_variable(lb=0., ub=1., name=f"z({i,j})") for i, j in unique_saddles]
        
        [model.add_linear_constraint(variables[i] == var_lb[i] + (var_ub[i] - var_lb[i]) * x[i], name=f"X_to_x({i})") 
         if variables_in_saddle_bool[i] else None for i in range(n)]
        
        [model.add_linear_constraint(Z[k] == var_lb[i] * var_lb[j] + var_lb[j] * (var_ub[i] - var_lb[i]) * x[i] + 
                                     var_lb[i] * (var_ub[j] - var_lb[j]) * x[j] + 
                                     (var_ub[j] - var_lb[j]) * (var_ub[i] - var_lb[i]) * z[k], name=f"Z_to_z({i},{j})") 
         for k, (i, j) in enumerate(unique_saddles)]

        if partition_method in ['Triangle', 'Square']:
            faces = list_faces_from_N(N, method=partition_method)
            list_coeffs, list_equations = equations_from_faces_3d(faces)
            COEFFS, EQUATIONS = [list_coeffs], [list_equations]
        else:
            list_coeffs_j, list_equations_j, list_coeffs_k, list_equations_k = equations_sum_convex(N)
            COEFFS = [list_coeffs_j, list_coeffs_k]
            EQUATIONS = [list_equations_j, list_equations_k]
        
        for k, (i, j) in enumerate(unique_saddles):
            add_pwl_constraints(model, x[i], x[j], z[k], COEFFS, EQUATIONS, logarithmic_encoding=logarithmic_encoding, name=f'{i},{j}')
    
    terms = []
    if quadratic:
        for i, j, a in saddle_list['objective']:
            terms.append(0.5 * a * variables[i] * variables[j])
    else:
        for i, j, a in saddle_list['objective']:
            k = dict_unique_saddles[(i, j)]
            terms.append(0.5 * a * Z[k])
    
    if obj_sense == 'MINIMIZE':
        model.minimize(sum([obj_lin[i] * variables[i] for i in range(n)]) + mathopt.fast_sum(terms))
    else:
        model.maximize(sum([obj_lin[i] * variables[i] for i in range(n)]) + mathopt.fast_sum(terms))

    if quadratic:
        for k in range(m):
            terms = []
            if saddle_list['constraints']:
                for i, j, a in saddle_list['constraints'][k]:
                    terms.append(0.5 * a * variables[i] * variables[j])
            model.add_quadratic_constraint((cons_lb[k] <= sum([cons_lin[k, j] * variables[j] for j in range(n)]) + mathopt.fast_sum(terms)) <= cons_ub[k])
    else:
        for k in range(m):
            terms = []
            if saddle_list['constraints']:
                for i, j, a in saddle_list['constraints'][k]:
                    p = dict_unique_saddles[(i, j)]
                    terms.append(0.5 * a * Z[p])
            model.add_linear_constraint((cons_lb[k] <= sum([cons_lin[k, j] * variables[j] for j in range(n)]) + mathopt.fast_sum(terms)) <= cons_ub[k])
    
    return model