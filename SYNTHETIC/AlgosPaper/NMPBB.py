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
def solve_W_NMPBB(A, C, W0, max_iter=1000, tol=1e-5):
    B = W0.copy()
    m, r = B.shape
    
    VHT = A @ C.T
    HHT = C @ C.T
    tr1 = np.sum(A * A)
    
    L = np.linalg.norm(HHT)
    if L == 0.0 or np.isnan(L) or np.isinf(L):
        L = 1.0
    
    alpha_max = 1e20
    alpha_min = 1e-20
    alpha_t = 1.0
    rho = 0.25
    eta0=.1
    gamma = 1e-4
    mu = 0.5
    c = 1e-4
    s = 1.7 
    
    S_val = 0.0
    f_Z_prev = 0.0
    # tol=1e-8
    iter=0
    for t in range(max_iter):
        # gradW = (B @ C - A) @ C.T
        gradW = B @ HHT - VHT
        iter=iter+1
        # if np.linalg.norm(proj(B - gradW) - B) < tol or f_Z_prev > 3e4: # use this only for high order problems
        if np.linalg.norm(proj(B - gradW) - B) < tol:
            break
            
        Z_t = proj(B - (1.0 / L) * gradW)
        gradZ = Z_t @ HHT - VHT
        # gradZ = (Z_t @ C - A) @ C.T
        f_Z = 0.5 * (tr1 - 2.0 * np.sum(Z_t * VHT) + np.sum((Z_t.T @ Z_t) * HHT))
        
        if t == 0:
            S_val = f_Z
        else:
            eta_t = (2.0 / np.pi) * np.arctan(abs(f_Z - f_Z_prev))
            S_val = f_Z + eta_t * (S_val - f_Z)
        f_Z_prev = f_Z
        
        # --- FIX: Replace boolean mask indexing with explicit loops ---
  # ------------------------------------------------------------
        # ACTIVE SET DIRECTION (Vectorized, Numba-safe)
        # ------------------------------------------------------------
        alpha_inv = 1.0/alpha_t
        
        # Boolean masks (Numba creates these efficiently)
        mask_active = Z_t <= alpha_inv * gradZ
        mask_L1 = mask_active & (gradZ >= c)
        mask_not_L1 = ~mask_L1
        
        # Generic projected BB step for free set / L2
        BB_step = proj(Z_t - alpha_t * gradZ) - Z_t
        
        # 1. Assign BB_step to not L1, and 0 to L1
        D_t = np.where(mask_not_L1, BB_step, 0.0)
        
        # 2. Overwrite L1^+ with -Z_t
        mask_L1_plus = mask_L1 & (Z_t != 0.0)
        D_t = np.where(mask_L1_plus, -Z_t, D_t)
        # ------------------------------------------------------------
        # Nonmonotone line search
        lambda_t = 1.0
        j=0.
        while True:
            W_candidate = Z_t + s * lambda_t * D_t
            f_Wcand = 0.5 * (tr1 - 2.0 * np.sum(W_candidate * VHT) + np.sum((W_candidate.T @ W_candidate) * HHT))
            S_candidate = f_Wcand + (0.5 if t == 0 else eta_t) * (S_val - f_Wcand)
            condition = np.sum(gradZ * D_t) + (mu / alpha_t) * np.sum(D_t * D_t)
            if S_candidate <= S_val + gamma * lambda_t * condition:
                break
            lambda_t *= rho
            j = j+1
            if lambda_t < 1e-20:
                break
        iter=iter+j    
        W_new = Z_t + s * lambda_t * D_t
        X_t = W_new - Z_t
        # Y_t = ((W_new @ C - A) @ C.T) - gradZ
        Y_t = (W_new @ HHT - VHT) - gradZ
        xy = np.sum(X_t * Y_t)
        if xy <= 0.0:
            alpha_t = alpha_max
        else:
            xx = np.sum(X_t * X_t)
            alpha_t = min(alpha_max, max(alpha_min, xx / xy))
            
        # W_new = np.where(np.isnan(W_new), 0.0, W_new)
        # W_new = np.where(np.isinf(W_new), 1e10, W_new)
        B = W_new
    return B, iter