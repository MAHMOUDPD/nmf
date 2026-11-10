
import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit
from statistics import mean, median
from scipy.interpolate import interp1d

@njit(cache=True, fastmath=True)
def adap_switch(alpha_t,tau_t):
    if tau_t >= alpha_t:
        alpha_cand=min(2*alpha_t, (1+(1/tau_t))*alpha_t)
    elif alpha_t/2 < tau_t < alpha_t:
        alpha_cand = tau_t
    elif 0< tau_t <= alpha_t/2:
        alpha_cand=np.sqrt((alpha_t*tau_t)/2.0)
    return alpha_cand

# --- Helper functions (Compiled with Numba) ---
@njit(cache=True, fastmath=True)
def proj(X):
    return np.maximum(X, 0.0)

@njit(cache=True, fastmath=True)
def solve_W_AdaPGE1(A, C, W0, max_iter=1000, tol=1e-5):
    B = W0.copy()
    m,r = B.shape
    
    HHT = C @ C.T
    VHT = A @ C.T
    # L = np.linalg.norm(HHT)
    L =np.sqrt(np.sum(HHT*HHT)) # O(r^2)
    if L == 0.0 or np.isnan(L) or np.isinf(L):
        L = 1.0
    
    alpha_t = 1./L
    # alpha_t = 1./L
    # alpha_t = 1e-10
    c = 1e-4
    # tol=1e-8
    
    alpha_min=1/(np.sqrt(2.0)*L)
    alpha_max=2/L

    mask_active_prev=np.zeros((m,r),dtype=np.bool_)
    stable_count=0
    # iter=0
    for t in range(max_iter):
        gradW = B @ HHT - VHT
        gradnorm=np.linalg.norm(proj(B - gradW) - B)
        if gradnorm < tol:
            break

        Z_t = proj(B - (1./L) * gradW)
        # Z_t = proj(B - alpha_t*gradW) ## aim at removing the dependence on L, but it seems to be worse than using 1/L
        gradZ = Z_t @ HHT - VHT
  # ------------------------------------------------------------
        # ACTIVE SET DIRECTION (Vectorized, Numba-safe)
        # ------------------------------------------------------------
        alpha_inv = alpha_t
        ## Boolean masks (Numba creates these efficiently)
        mask_active = (Z_t <= alpha_inv * gradZ)
        mask_L1 = mask_active & (gradZ >= c)
        mask_L2 = ~mask_L1
        mask_free = ~mask_active
        mask_L1_plus = mask_L1 & (Z_t != 0.0)

        # Generic projected BB step for free set / L2  
        ## D_t = proj(Z_t - alpha_t * gradZ) - Z_t      
        BB_step = proj(Z_t - alpha_t*gradZ) - Z_t 
        # 1. Assign BB_step to not L1, and 0 to L1
        D_t = np.where(mask_free|mask_L2, BB_step, 0.0)
        D_t = np.where(mask_L1_plus, -Z_t, D_t)

        # VERSION. 3: Use ADAPTIVE step size (s) for simplicity
        # 1. Compute a = < gradZ, D_t > (We already have gradZ and D_t)
        a = np.sum(gradZ*D_t)
        # 2. Compute b = ||D_t C||_F^2 using Trace( (D_t.T @ D_t) * (C @ C.T) )
        # HHT = C @ C.T was already computed at the start of the function
        DtDt = D_t.T@D_t              # O(m*r^2)
        b = np.sum(DtDt*HHT)          # O(r^2) - completely avoids D_t @ C (O(mnr))
        
        if b < 1e-12:
            lambda_t = 1.0
        else:
            lambda_exact = -a / b       # The true optimal step
            # OPTMIZATION: Skip lambda_max loop when boundary is far
            min_Z=np.min(Z_t)
            boundary_thres=1e-5
            # ONLY compute the expensive min ratio of Z_t is close zero
            if min_Z >boundary_thres:
                # bound ius maths guarantee to be >1
                # since exact step is typically ~1/L, we avoid the O(mr) loop entirely
                lambda_max=1e20
            else:
                #  compute lambda_max when variables are near the bouddary
                lambda_max=1e20
            # if min_Z<= boundary_thres:
                mask_neg=D_t < 0.0
                if np.any(mask_neg):
                   ## Compute ratio only where D_t is negative; fill 1e20 everywhere else
                    ratio = np.where(mask_neg, -Z_t/D_t, lambda_max)
                    lambda_max = np.min(ratio)
                    
            # Feasible step taking the nonnegativity constraint into account
            lambda_t = lambda_exact if lambda_exact < lambda_max else lambda_max
            if lambda_t < 0.0:
                lambda_t = 0.0
        # ------------------------------------------------------------
        S_t=lambda_t*D_t
        W_new = Z_t + S_t
        # --------------------------------------------------------------
        # Check early termination via active-set stability
        # --------------------------------------------------------------
        # ----------------------------------------------------------
        # active_changed=False
        # if t>0:
        #     for i in range(m):
        #         for j in range(r):
        #             if mask_active[i,j]!=mask_active_prev[i,j]:
        #                 active_changed=True
        #                 break
        #         if active_changed:
        #             break
        # if active_changed:
        #     stable_count=0
        # else:
        #     # if active set hasn't changed and displacement is tiny, we are done
        #     displacement_norm=np.linalg.norm(S_t)
        #     if displacement_norm<1e-5:
        #         stable_count+=1
        #     else:
        #         stable_count=0
        # # if stable for 5 iterations, break early
        # if stable_count>=5:
        #     break
        # # prepare for next iteration
        # mask_active_prev=mask_active.copy()
        # ----------------------------------------------------------
        # tau_t= lambda_t # I used this in experiments!
        tau_t=lambda_exact
        alpha_cond=adap_switch(alpha_t,tau_t)
        alpha_next = max(alpha_min, min(alpha_max, alpha_cond))
        alpha_t=alpha_next

        B = W_new
    return B, t+1




import numpy as np

def compute_restricted_condition_number(A, X_final, tol=1e-8):
    """
    A: The fixed factor matrix (e.g., B or C) of the NMF subproblem.
    X_final: The optimal solution of the subproblem (e.g., H_star or W_star).
    tol: Threshold to determine if a variable is 'free' (positive).
    """
    # 1. Identify free indices (strictly positive values)
    free_indices = np.where(X_final > tol)[0]
    
    # 2. Compute the full Hessian
    H_full = A.T @ A
    
    # 3. Extract the restricted Hessian
    # Note: For dense matrices, use np.ix_ for clean submatrix extraction
    H_restricted = H_full[np.ix_(free_indices, free_indices)]
    
    # 4. Compute Eigenvalues to avoid SVD when matrix is square, or just use svd
    eigvals = np.linalg.eigvalsh(H_restricted)
    
    # 5. Compute Kappa
    lambda_min = np.min(eigvals)
    lambda_max = np.max(eigvals)
    
    # Safety check to avoid division by zero if severely ill-conditioned
    if lambda_min < 1e-12:
        return np.inf
        
    kappa = lambda_max / lambda_min
    return kappa, lambda_min, lambda_max