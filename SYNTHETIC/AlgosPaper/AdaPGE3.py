
import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit
from statistics import mean, median
from scipy.interpolate import interp1d
# from Utils import *
# --- Helper functions (Compiled with Numba) ---

@njit(cache=True, fastmath=True)
def adap_switch(alpha_t,tau_t):
    if tau_t >= alpha_t:
        alpha_cand=min(2*alpha_t, (1+(1/tau_t))*alpha_t)
    elif alpha_t/2 < tau_t < alpha_t:
        alpha_cand = tau_t
    elif 0< tau_t <= alpha_t/2:
        alpha_cand=np.sqrt((alpha_t*tau_t)/2.0)
    return alpha_cand

@njit(cache=True, fastmath=True)
def proj(X):
    return np.maximum(X, 0.0)

@njit(cache=True, fastmath=True)
def solve_W_AdaPGE3(A, C, W0, max_iter=1000, tol=1e-5):
    B = W0.copy()
    m, r = B.shape
    
    HHT = C @ C.T
    VHT = A @ C.T

    alpha_t = 1.
    # alpha_t = 1./L
    # alpha_t = 1e-10
    beta = 1e-4
    s = 1.0
    # tol=1e-8
    gradW = B @ HHT - VHT
    # gradW =np.zeros_like(B)

    mask_active_prev=np.zeros((m,r),dtype=np.bool_)
    stable_count=0
    # iter=0
    for t in range(max_iter):
        if np.linalg.norm(proj(B-gradW)-B)<tol:
            break

        Z_t = proj(B - alpha_t*gradW)
        # Z_t = proj(B - alpha_t*gradW) # aim at removing the dependence on L, but it seems to be worse than using 1/L
        gradZ = Z_t @ HHT - VHT
  # ------------------------------------------------------------
        # ACTIVE SET DIRECTION (Vectorized, Numba-safe)
        # ------------------------------------------------------------
        alpha_inv = alpha_t
        ## Boolean masks (Numba creates these efficiently)
        mask_active = (Z_t <= alpha_inv * gradZ)
        mask_L1 = mask_active & (gradZ >= beta)
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
        a = np.sum(gradZ * D_t)
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
                    ratio = np.where(mask_neg, -Z_t/D_t, 1e20)
                    lambda_max = np.min(ratio)
                    
            # Feasible step taking the nonnegativity constraint into account
            lambda_t = lambda_exact if lambda_exact < lambda_max else lambda_max
            if lambda_t < 0.0:
                lambda_t = 0.0
        # ------------------------------------------------------------
        S_t=s*lambda_t*D_t
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
        # -----------------------------------------------------------
        ## Get BB Short: USE IN AdaPGE2,AdaPGE3
        # xx=(lambda_t**2)*np.sum(DtDt)
        # X_t=W_new-Z_t
        gradW_next=W_new@HHT - VHT
        Y_t = gradW_next- gradZ
        yy = np.sum(Y_t*Y_t)
        sy = np.sum(S_t*Y_t)

        thres=1e-12
        tau_bb_next=max(thres, sy/yy)

        alpha_min=tau_bb_next/2.0
        alpha_max=2*tau_bb_next
        # -----------------------------------
        alpha_cond=adap_switch(alpha_t,tau_bb_next)
        alpha_next = max(alpha_min, min(alpha_max, alpha_cond))
        alpha_t=alpha_next
        gradW=gradW_next.copy()
        B = W_new
        # gradnorm=np.linalg.norm(proj(B - gradW) - B)
        # if gradnorm < tol:
        #     break
    return B, t+1