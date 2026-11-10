import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit
from statistics import mean, median
from scipy.interpolate import interp1d

# ---------------------------------------------------------
# Algorithm 1: Improved Monotone Projected BB Step Size (Subproblem Solver)
# ---------------------------------------------------------
@njit(cache=True, fastmath=True)
def proj(X):
    return np.maximum(X, 0.0)

@njit(cache=True, fastmath=True)
def solve_EPMPBBW(V, A, X0, max_iter=200, tol=1e-5):
    """
    Solves min_{X >= 0} 0.5 * || V - A X ||_F^2
    
    V: m x n matrix
    A: m x r matrix (Fixed factor)
    X0: r x n initial guess
    """
    X = X0.copy()
    m, r = A.shape
    # Precompute constants for trace-based operations
    AtA = A.T @ A                   # r x r
    AtV = A.T @ V                   # r x n
    tr1 = np.sum(V * V)
    
    # Lipschitz constant
    L = np.linalg.norm(AtA, 2)
    if L == 0.0 or np.isnan(L) or np.isinf(L):
        L = 1.0
    
    sigma = 0.1
    alpha_max = 1e20
    alpha_min = 1e-20
    theta = 0.6          # Weighted BB parameter (Section IV.A)
    tau = 0.5       # Adaptive selection threshold (Section IV.B)
    alpha = 1.0
    iter=0.0
    # tol=1e-8
    for t in range(max_iter):
        # Gradient w.r.t X: A^T (A X - V) = AtA @ X - AtV
        grad = AtA @ X - AtV
        iter=iter+1
        # Convergence check (Projected gradient step)
        if np.linalg.norm(proj(X - grad) - X) < tol:
            break
        
        # Step 3: Proximal point
        Z = proj(X - (1.0 / L) * grad)
        
        # Gradient at Z
        gradZ = AtA @ Z - AtV
        
        # Step 4: Direction
        D = proj(Z - alpha * gradZ) - Z
        
        # Step 5: Calculate step size rho
        # delta = < D, (A^T A) D > = || A D ||_F^2
        AD = A @ D
        delta = np.sum(AD * AD)
        
        if delta < 1e-12:
            rho = 1.0
        else:
            # mu = -(1 - sigma) < gradZ, D > / delta
            mu = -(1.0 - sigma) * np.sum(gradZ * D) / delta
            rho = min(max(mu, 0.0), 1.0)  # Clamp between 0 and 1
        
        # Update X
        X_new = Z + rho * D
        
        # Step 6: Calculate BB step size (Formula 7, BB1)
        grad_new = AtA @ X_new - AtV
        S = X_new - Z
        Y = grad_new - gradZ
        
        sy = np.sum(S * Y)
        ss = np.sum(S * S)
        yy = np.sum(Y * Y)
        
        if sy <= 1e-12:
            alpha = alpha_max
        else:
            # Compute standard BB steps
            alpha_BB1 = ss / sy
            
            if yy <= 1e-12:
                alpha_BB2 = alpha_BB1  # Fallback
            else:
                alpha_BB2 = sy / yy
            
            # Compute Weighted BB (WBB)
            num = (1.0 - theta) * alpha_BB1 + 2.0 * theta
            den = (1.0 - theta) * alpha_BB2 + 2.0 * theta
            if den < 1e-12:
                alpha_WBB = alpha_BB1
            else:
                alpha_WBB = alpha_BB2 * (num / den)
            
            # Adaptive Selection (AWBB)
            if alpha_BB2 / alpha_BB1 < tau:
                alpha = alpha_BB2
            else:
                alpha = alpha_WBB
            
            # Safeguard
            alpha = min(alpha_max, max(alpha_min, alpha))
        
        X = X_new
        
    return X, iter