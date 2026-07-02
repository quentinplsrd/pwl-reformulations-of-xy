# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog

def N_from_target_error(target_error):
    return int(np.ceil(1 / (4 * np.sqrt(target_error))))

def list_faces_from_N(N, method='Triangle'):
    faces = []
    eps = 1e-3 / N
    for j in range(2 * N):
        for k in range(2 * N):
            x = (j - k) / (2 * N) + 0.5
            y = (j + k + 1) / (2 * N) - 0.5
            if (x >= 0) and (x <= 1) and (y >= 0) and (y <= 1):
                pts = []
                for delta in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    xx = x + delta[0] / (2 * N)
                    yy = y + delta[-1] / (2 * N)
                    if (xx >= 0 - eps) and (xx <= 1 + eps) and (yy >= 0 - eps) and (yy <= 1 + eps):
                        pts.append([xx, yy, xx * yy])
                pts = np.array(pts)
                faces.append(pts)
                
    if method == 'Triangle':
        faces_triang = []
        for face in faces:
            if len(face) == 3:
                faces_triang.append(face)
            else:
                x, y = face.mean(axis=0)[0:2]
                if (y >= x - eps) ^ (x + y >= 1 - eps):
                    face = face[face[:, 0].argsort(), :]
                else:
                    face = face[face[:, 1].argsort(), :]
                faces_triang.append(face[0:3, :])
                faces_triang.append(face[1:4, :])
        faces = list(faces_triang)
    return faces

def list_faces_from_N_DC(N):
    faces_plus = []
    faces_minus = []
    for j in range(2 * N):
        pts = []
        if j == 0:
            pts.append([0, 0, evaluate_gn(np.array([[0, 0]]), N)[0][0]])
        elif j < N:
            pts.append([0, j / N, evaluate_gn(np.array([[0, j / N]]), N)[0][0]])
            pts.append([j / N, 0, evaluate_gn(np.array([[j / N, 0]]), N)[0][0]])
        else:
            pts.append([(j - N) / N, 1, evaluate_gn(np.array([[(j - N) / N, 1]]), N)[0][0]])
            pts.append([1, (j - N) / N, evaluate_gn(np.array([[1, (j - N) / N]]), N)[0][0]])
            
        if j == 2 * N - 1:
            pts.append([1, 1, evaluate_gn(np.array([[1, 1]]), N)[0][0]])
        elif j < N:
            pts.append([(j + 1) / N, 0, evaluate_gn(np.array([[(j + 1) / N, 0]]), N)[0][0]])
            pts.append([0, (j + 1) / N, evaluate_gn(np.array([[0, (j + 1) / N]]), N)[0][0]])
        else:
            pts.append([1, (j + 1 - N) / N, evaluate_gn(np.array([[1, (j + 1 - N) / N]]), N)[0][0]])
            pts.append([(j + 1 - N) / N, 1, evaluate_gn(np.array([[(j + 1 - N) / N, 1]]), N)[0][0]])
        pts = np.array(pts)
        faces_plus.append(pts)
        
    for k in range(2 * N):
        pts = []
        if k == 0:
            pts.append([1, 0, evaluate_gn(np.array([[1, 0]]), N)[1][0]])
        elif k < N:
            pts.append([1 - k / N, 0, evaluate_gn(np.array([[1 - k / N, 0]]), N)[1][0]])
            pts.append([1, k / N, evaluate_gn(np.array([[1, k / N]]), N)[1][0]])
        else:
            pts.append([0, (k - N) / N, evaluate_gn(np.array([[0, (k - N) / N]]), N)[1][0]])
            pts.append([1 - (k - N) / N, 1, evaluate_gn(np.array([[1 - (k - N) / N, 1]]), N)[1][0]])
            
        if k == 2 * N - 1:
            pts.append([0, 1, evaluate_gn(np.array([[0, 1]]), N)[1][0]])
        elif k < N:
            pts.append([1, (k + 1) / N, evaluate_gn(np.array([[1, (k + 1) / N]]), N)[1][0]])
            pts.append([1 - (k + 1) / N, 0, evaluate_gn(np.array([[1 - (k + 1) / N, 0]]), N)[1][0]])
        else:
            pts.append([1 - (k + 1 - N) / N, 1, evaluate_gn(np.array([[1 - (k + 1 - N) / N, 1]]), N)[1][0]])
            pts.append([0, (k + 1 - N) / N, evaluate_gn(np.array([[0, (k + 1 - N) / N]]), N)[1][0]])
        pts = np.array(pts)
        faces_minus.append(pts)
        
    return faces_plus, faces_minus

def evaluate_gn(x_y_array, N):
    j_vec = np.arange(2 * N)
    k_vec = np.arange(2 * N)
    mat_plus = np.c_[(2 * j_vec + 1) / (4 * N), (2 * j_vec + 1) / (4 * N), - (j_vec * (j_vec + 1)) / (4 * N**2)]
    mat_minus = np.c_[-(2 * (k_vec - N) + 1) / (4 * N), (2 * (k_vec - N) + 1) / (4 * N), - (k_vec * (k_vec + 1) - 2 * N * k_vec - N * (1 - N)) / (4 * N**2)]
    x_y_1 = np.insert(x_y_array, 2, 1, axis=1)
    z_plus = np.max(np.matmul(mat_plus, x_y_1.T), axis=0)
    z_minus = np.max(np.matmul(mat_minus, x_y_1.T), axis=0)
    return z_plus, z_minus

def equations_sum_convex(N, decimal_tol=9):
    list_coeffs_j, list_equations_j = [], []
    list_coeffs_k, list_equations_k = [], []
    
    for j in range(2 * N):
        tmp_eq = []
        if j > 0:
            tmp_eq.append([-1, -1, j / N])
        if j < 2 * N - 1:
            tmp_eq.append([1, 1, -(j + 1) / N])
        tmp_eq = np.array(tmp_eq)
        list_equations_j.append(np.round(tmp_eq, decimal_tol))
        a = b = (2 * j + 1) / (4 * N)
        c = -j * (j + 1) / (4 * N**2)
        list_coeffs_j.append(np.round(np.array([a, b, c]), decimal_tol))
        
    for k in range(2 * N):
        tmp_eq = []
        if k < 2 * N - 1:
            tmp_eq.append([-1, 1, 1 - (k + 1) / N])
        if k > 0:
            tmp_eq.append([1, -1, -1 + k / N])
        tmp_eq = np.array(tmp_eq)
        list_equations_k.append(np.round(tmp_eq, decimal_tol))
        a = (2 * (k - N) + 1) / (4 * N)
        b = -(2 * (k - N) + 1) / (4 * N)
        c = (k - N) * (k - N + 1) / (4 * N**2)
        list_coeffs_k.append(np.round(np.array([a, b, c]), decimal_tol))
        
    return list_coeffs_j, list_equations_j, list_coeffs_k, list_equations_k

def equations_from_faces_3d(faces, decimal_tol=9):
    list_coeffs, list_equations = [], []
    for face in faces:
        M = face[:3, :] * 1
        M[:, 2] = 1.
        Z = face[:3, 2] * 1
        coeffs = np.linalg.solve(M, Z)
        list_coeffs.append(np.round(coeffs, decimal_tol))
        hull_2d = ConvexHull(face[:, :2])
        list_equations.append(np.round(hull_2d.equations, decimal_tol))
        
    return list_coeffs, list_equations

def faces_from_equations(list_coeffs, list_equations, x_lb=None, x_ub=None, y_lb=None, y_ub=None):
    extra_halfspaces = np.zeros((0, 3))
    if x_lb is not None: extra_halfspaces = np.r_[extra_halfspaces, [[-1, 0, x_lb]]]
    if x_ub is not None: extra_halfspaces = np.r_[extra_halfspaces, [[1, 0, -x_ub]]]
    if y_lb is not None: extra_halfspaces = np.r_[extra_halfspaces, [[0, -1, y_lb]]]
    if y_ub is not None: extra_halfspaces = np.r_[extra_halfspaces, [[0, 1, -y_ub]]]
    
    faces = []
    for k in range(len(list_coeffs)):
        coeffs = list_coeffs[k]
        halfspaces = list_equations[k]
        if halfspaces is None:
            faces.append(None)
            continue
        halfspaces = np.r_[halfspaces, extra_halfspaces]
        x = find_interior_point(halfspaces)
        hs = HalfspaceIntersection(halfspaces, x)
        hull = ConvexHull(hs.intersections)
        xy = hs.intersections[hull.vertices]
        z = np.insert(xy, 2, 1, axis=1) @ coeffs.reshape(-1, 1)
        face = np.c_[xy, z]
        faces.append(face)
        
    return faces

def find_interior_point(halfspaces):
    x = None
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.c_[halfspaces[:, :-1], norm_vector]
    b = - halfspaces[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)])
    if res.success:
        x = res.x[:-1]
    return x