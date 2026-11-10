from matplotlib.pylab import beta
import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit
from statistics import mean, median
from scipy.interpolate import interp1d

# --- Helper functions (Compiled with Numba) ---
@njit(cache=True, fastmath=True)
def proj(X):
    return np.maximum(X, 0.0)

@njit(cache=True, fastmath=True)
def compute_kkp_proj_norm(B, C, A):
    gradW = (B @ C - A) @ C.T
    gradH = B.T @ (B @ C - A)
    pgradW = gradW.copy()
    pgradH = gradH.copy()
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if B[i, j] == 0.0 and gradW[i, j] > 0.0:
                pgradW[i, j] = 0.0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] == 0.0 and gradH[i, j] > 0.0:
                pgradH[i, j] = 0.0
    return np.sqrt(np.sum(pgradW**2) + np.sum(pgradH**2))

# --- Evaluation Metric ---
def rel_residual(A, B, C):
    return np.linalg.norm(A - B @ C, 'fro') / np.linalg.norm(A, 'fro')


def solve_NMF_ANLS(A, r, solver_func, max_outer=50, inner_tol=1e-4, inner_max=200, epsilon=1e-8):
    m, n = A.shape
    B = np.maximum(0, np.random.randn(m, r))
    C = np.maximum(0, np.random.randn(r, n))

    def compute_obj(A, W_matrix, H_matrix):
        tr_V = np.sum(A*A)
        VHT = A @ H_matrix.T
        HHT = H_matrix @ H_matrix.T
        return 0.5 * (tr_V - 2.0 * np.sum(W_matrix * VHT) + np.sum((W_matrix.T @ W_matrix) * HHT))

    start_time = time.time()
    e0 = compute_obj(A, B, C)   
    pg0_norm = compute_kkp_proj_norm(B, C, A)
    if pg0_norm < 1e-12: 
        return B, C, 0.0, 0.0, [0.0], [0.0], [rel_residual(A, B, C)], [0.0],[pg0_norm]
    
    # Initialize histories with the initial state
    obj_hist = [e0]
    res_hist = [rel_residual(A, B, C)]  # <--- NEW
    time_hist = [0.0]
    iter_hist = [0.0]
    proj_hist =  [pg0_norm]
    # iter_count=[0.0]
        # C = solve_OGM_subproblem(A, B, C, max_iter=inner_max, tol=inner_tol)

        # B = solve_OGM_subproblem(A.T, C.T, B.T, max_iter=inner_max, tol=inner_tol).T

    for k in range(max_outer):
        
        B,it1 = solver_func(A, C, B, max_iter=inner_max, tol=inner_tol)
        # B = proj(B)        
        C,it2 = solver_func(A.T, B.T, C.T, max_iter=inner_max, tol=inner_tol)
        
        C=C.T
        # C = proj(C)
        elapsed = time.time()-start_time
        current_obj = compute_obj(A, B, C)
        current_res = rel_residual(A, B, C) # <--- NEW
        
        obj_hist.append(current_obj)
        res_hist.append(current_res)        # <--- NEW
        time_hist.append(elapsed)

        iter_hist.append(it1+it2)
        
        pg_k_norm = compute_kkp_proj_norm(B, C, A)
        proj_hist.append(pg_k_norm)

        if pg_k_norm <= epsilon * pg0_norm:
            break
            
    return B, C, time_hist[-1],proj_hist[-1],iter_hist,obj_hist, res_hist, time_hist,proj_hist

def solve_NMF_ANLS2(A, r, solver_func, max_outer=50, inner_tol=1e-4, inner_max=200, epsilon=1e-8):
    m, n = A.shape
    B = np.maximum(0, np.random.randn(m, r))
    C = np.maximum(0, np.random.randn(r, n))

    def compute_obj(A, W_matrix, H_matrix):
        tr_V = np.sum(A*A)
        VHT = A @ H_matrix.T
        HHT = H_matrix @ H_matrix.T
        return 0.5 * (tr_V - 2.0 * np.sum(W_matrix * VHT) + np.sum((W_matrix.T @ W_matrix) * HHT))

    start_time = time.time()
    e0 = compute_obj(A, B, C)   
    pg0_norm = compute_kkp_proj_norm(B, C, A)
    if pg0_norm < 1e-12: 
        return B, C, 0.0, 0.0, [0.0], [e0], [rel_residual(A, B, C)], [0.0],[pg0_norm]
    
    # Initialize histories with the initial state
    obj_hist = [e0]
    res_hist = [rel_residual(A, B, C)]  # <--- NEW
    time_hist = [0.0]
    iter_hist = [0.0]
    proj_hist =  [pg0_norm]
    H0=C.copy()
    W0=B.copy()
    beta0=.99

    eps_W = max(1e-3, epsilon)*pg0_norm
    eps_H = max(1e-3, epsilon)*pg0_norm
    K_min =10  # Minimum number of inner iterations before reducing tolerance`

    for k in range(max_outer):
        B,it1 = solver_func(A, C, B, max_iter=inner_max, tol=eps_W)
        if it1 < K_min:
            eps_W *= 0.1

        C,it2 = solver_func(A.T, B.T, C.T, max_iter=inner_max, tol=eps_H)
        C=C.T
        if it2 < K_min:
            eps_H *= 0.1
        elapsed = time.time() - start_time
        current_obj = compute_obj(A, B, C)
        current_res = rel_residual(A, B, C) # <--- NEW
        
        obj_hist.append(current_obj)
        res_hist.append(current_res)        # <--- NEW
        time_hist.append(elapsed)

        iter_hist.append(it1+it2)
        # H0=C.copy()
        # W0=B.copy()
        pg_k_norm = compute_kkp_proj_norm(B, C, A)
        proj_hist.append(pg_k_norm)
        # beta= pg_k_norm/pg0_norm
        if pg_k_norm <= epsilon*pg0_norm:
            break
        # beta0=beta
        # pg0_norm=pg_k_norm
    return B, C, time_hist[-1],proj_hist[-1],iter_hist,obj_hist, res_hist, time_hist,proj_hist

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def svd_initialization(A, r):
    """SVD-based initialization as described in Algorithm 2 Step 1"""
    # Numba cannot SVD, so we do this in standard Python/Numpy
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    B = U[:, :r] * np.sqrt(S[:r])
    C = np.sqrt(S[:r])[:, None] * Vt[:r, :]
    # Ensure nonnegativity
    B = np.maximum(B, 0.0)
    C = np.maximum(C, 0.0)
    return B, C

def compute_obj(A, B, C):
    """Trace-based objective function evaluation"""
    tr1 = np.sum(A * A)
    VHT = A @ C.T
    HHT = C @ C.T
    return 0.5 * (tr1 - 2.0 * np.sum(B * VHT) + np.sum((B.T @ B) * HHT))

# ---------------------------------------------------------
# Algorithm 2: EPMPBB Main Framework
# ---------------------------------------------------------
def solve_NMF_EPMPBB(A, r,solver_func, max_outer=100, inner_tol=1e-5, inner_max=200, outer_tol=1e-8):
    """
    Extrapolated Monotone Projected Barzilai-Borwein NMF.
    """
    m, n = A.shape
    # Step 1: SVD Initialization
    start_time = time.time()

    B, C = svd_initialization(A, r)
    
    # Step 2: Initialize extrapolation variables
    W_y = B.copy()
    H_y = C.copy()
    beta = 0.5
    gamma = 1.1
    eta = 1.5
    
    e0 = compute_obj(A, B, C)
    e_prev = e0
    
    # start_time = time.time()
    W_prev_outer = B.copy()  # Used for pullback logic
    pg0_norm = compute_kkp_proj_norm(B, C, A)
    if pg0_norm < 1e-12: 
        return B, C, 0.0, 0.0, [0.0], [e0], [rel_residual(A, B, C)], [0.0],[pg0_norm]
    
    start_time = time.time()
    
    # Initialize histories with the initial state
    obj_hist = [e0]
    res_hist = [rel_residual(A, B, C)]  # <--- NEW
    time_hist = [0.0]
    iter_hist = [0.0]
    proj_hist =  [pg0_norm]  

    eps_W = max(1e-3, outer_tol)*pg0_norm
    eps_H = max(1e-3, outer_tol)*pg0_norm
    K_min =10  # Minimum number of inner iterations before reducing tolerance`


    for k in range(max_outer):
        # Step 3: Fix W_y and update C
        # Transposition logic: A -> A, Fixed -> W_y, Variable -> H_y
        H_n,it1 = solver_func(A, W_y, H_y, max_iter=inner_max, tol=eps_W)
        if it1 < K_min:
            eps_W *= 0.1
        # Step 4: Extrapolate H_y and project
        H_y = H_n + beta*(H_n - C)
        H_y = proj(H_y)
        
        # Step 5: Fix H_y and update B
        # Transposition logic: A -> A.T, Fixed -> H_y.T, Variable -> W_y.T
        W_n,it2 = solver_func(A.T, H_y.T, W_y.T, max_iter=inner_max, tol=eps_H)
        W_n = W_n.T
        if it2 < K_min:
            eps_H *= 0.1
        # Step 6: Extrapolate W_y and project
        W_y = W_n + beta * (W_n - B)
        W_y = proj(W_y)
        
        # Step 7: Evaluate error and perform pullback/extrapolation updates
        e_curr = compute_obj(A, W_n, H_y)
        
        if e_curr > e_prev:
            # Pullback and reset extrapolation
            H_y = H_n
            W_y = W_prev_outer  # Revert to previous accepted B
            beta = beta / eta   # Reduce extrapolation speed
            # Keep B and C unchanged, do not update e_prev
        else:
            # Accept the update
            W_prev_outer = W_n
            B = W_n
            C = H_y
            e_prev = e_curr
            beta = min(1.0, gamma * beta)  # Accelerate extrapolation
        elapsed = time.time() - start_time
        
        time_hist.append(elapsed)       
        # current_obj = compute_obj(B, C)
        current_res = rel_residual(A, B, C) # <--- NEW
        
        obj_hist.append(e_curr)
        res_hist.append(current_res)        # <--- NEW

        iter_hist.append(it1+it2)
        
        pg_k_norm = compute_kkp_proj_norm(B, C, A)
        proj_hist.append(pg_k_norm)

        if pg_k_norm <= outer_tol*pg0_norm:
            break
            
    return B, C, time_hist[-1],proj_hist[-1],iter_hist,obj_hist, res_hist, time_hist,proj_hist

def compute_obj_NeNMF(A, B, C):
    """
    Exact Trace-based evaluation of F(B,C) = 0.5 * ||A - B C||_F^2
    Based on Eq (40) of the NeNMF paper.
    Complexity is O(nr^2) instead of O(mnr).
    """
    tr1 = np.sum(A * A)
    WTV = B.T @ A
    WtW = B.T @ B
    HHt = C @ C.T
    return 0.5 * (tr1 - 2.0 * np.sum(C * WTV) + np.sum(WtW * HHt))

@njit(cache=True, fastmath=True)
def compute_proj_grad_norm(A, B, C):
    """
    Computes the KKT projected gradient norm for BOTH B and C.
    Used for the outer stopping criterion (Eq 22).
    """
    gradW = (B @ C - A) @ C.T
    gradH = B.T @ (B @ C - A)
    pgradW = gradW.copy()
    pgradH = gradH.copy()
    
    # Projected gradient for B
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if B[i, j] == 0.0 and gradW[i, j] > 0.0:
                pgradW[i, j] = 0.0

    # Projected gradient for C
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] == 0.0 and gradH[i, j] > 0.0:
                pgradH[i, j] = 0.0

    return np.sqrt(np.sum(pgradW**2) + np.sum(pgradH**2))

def solve_NMF_NeNMF(A, r,solver_func, max_outer=100, inner_tol=1e-4, inner_max=200, outer_tol=1e-8):
    """
    NeNMF: An Optimal Gradient Method for Nonnegative Matrix Factorization.
    Specifically tuned for Nesterov's OGM with correct transposition logic.
    """
    m, n = A.shape
    
    # Random dense initialization (as per the paper's experiments)
    B = np.maximum(0, np.random.randn(m, r))
    C = np.maximum(0, np.random.randn(r, n))
    
    # Initial projected gradient norm (used for relative stopping Eq 22)
    pg0_norm = compute_proj_grad_norm(A, B, C)
    if pg0_norm < 1e-12:
        return B, C, 0.0, pg0_norm, 0, [compute_obj_NeNMF(A,B,C)], [rel_residual(A,B,C)], [0.0], [pg0_norm]
    
    start_time = time.time()
    
    eps_W = max(1e-3, outer_tol) * pg0_norm
    eps_H = max(1e-3, outer_tol) * pg0_norm
    
    K_min = 10  # The predefined minimum step number from the quote

    # Initialize history tracking
    obj_hist = [compute_obj_NeNMF(A, B, C)]
    res_hist = [rel_residual(A, B, C)]
    time_hist = [0.0]
    iter_hist = [0.0]
    proj_hist = [pg0_norm]
    total_inner_its = 0
    
    for k in range(max_outer):
        # Step 2.1: Fix B, update C 
        # (Standard format: A is A, A is B, X is C)
        C, it1 = solver_func(A, B, C, max_iter=inner_max, tol=eps_H)
        total_inner_its += it1
        if it1 < K_min:
            eps_H *= 0.1        
        # Step 2.2: Fix C, update B 
        # (CRITICAL CORRECTION: A -> A.T, A -> C.T, X -> B.T)
        B, it2 = solver_func(A.T, C.T, B.T, max_iter=inner_max, tol=eps_W)
        total_inner_its += it2
        B = B.T  # Transpose back to original dimensions
        
        # ADAPTIVE TOLERANCE CHECK FOR B
        if it2 < K_min:
            eps_W *= 0.1
        
        elapsed = time.time() - start_time
        current_obj = compute_obj_NeNMF(A, B, C)
        current_res = rel_residual(A, B, C)
        pg_k_norm = compute_proj_grad_norm(A, B, C)
        
        obj_hist.append(current_obj)
        res_hist.append(current_res)
        iter_hist.append(total_inner_its)

        time_hist.append(elapsed)
        proj_hist.append(pg_k_norm)
        
        # Step 3: Check global convergence (Eq 22)
        if pg_k_norm <= outer_tol*pg0_norm:
            break
            
    # elapsed = time.time() - start_time
    return B, C, time_hist[-1], pg_k_norm, iter_hist, obj_hist, res_hist, time_hist, proj_hist