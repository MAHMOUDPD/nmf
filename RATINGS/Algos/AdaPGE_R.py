
import numpy as np
import time
from numba import njit

# import Utils
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
def solve_AdaPGE_R(C, A, B0, N, mu, variant='AdaPGE1', max_iter=200, tol=1e-7):
    """
    Solve min_{B >= 0}  (1/2N) ||C - B A^T||_F^2 + 0.5*mu*||B||_F^2
    using the adaptive exact-step method.
    C : (m, n) centered data matrix  C = (s - mu) at observed entries
    A : (n, r) fixed factor matrix
    B0: (m, r) initial guess for the factor matrix B
    N : number of observed entries in C
    mu: regularization parameter
    """
    r = A.shape[0]
    B = B0.copy()

    AAt= A@A.T
    # AtX = A.T @ C
    H = AAt/N + mu*np.eye(r)                 # Hessian of the subproblem
    L=np.linalg.norm(H, 2)                     # Lipschitz constant of the gradient
    if L == 0.0 or np.isnan(L) or np.isinf(L):
        return B, 0.0
    CA=C@A.T
    grad_const = -CA/N                          # constant part of the gradient
    c=1e-4
    # bounds for the step size: fixed for AdaPGE1/2, dynamic for AdaPGE3    
    if variant != 'AdaPGE3':
        alpha_t=1.0/L
        alpha_max = 0.5/L
        alpha_min = 2.0/L
    else:
        alpha_min, alpha_max=None, None
        alpha_t=1.0

    it=0
    for t in range(max_iter):
        # ---gradient ---
        grad = B @ H + grad_const
        if not np.all(np.isfinite(grad)):
            print("Warning: Non-finite gradient encountered.")
            break
        # ----projected - gradient stopping criterion ---
        if np.linalg.norm(proj(B - grad) - B) < tol:
            break
        # --- proximal point step ---
        if variant !='AdaPGE3':
            Y=proj(B - (1.0/L)*grad)
        else:
            Y=proj(B - alpha_t * grad)
        # --- active set identificatiion ---
        grad_Y=Y@H + grad_const
        mask_active = (Y <= alpha_t*grad_Y)
        mask_L1 = mask_active & (grad_Y>= c)
        mask_L2 = ~mask_L1
        mask_L1_plus = mask_L1 &(Y!=0.0)
        mask_free = ~mask_active
        # --- direction P^t---
        # P=np.zeros_like(Y)
        # P[mask_L1_plus]=-Y[mask_L1_plus]
        # if np.any(mask_free):
        #     Z_cand=proj(Y-alpha_t*grad_Y)
        #     P[mask_free]=Z_cand[mask_free]-Y[mask_free]

        # OR USE THE FOLLOWING IMPL!
        # ------------------------------
        BB_step = proj(Y - alpha_t * grad_Y) - Y
        P = np.where(mask_free | mask_L2, BB_step, 0.0) 
        P=np.where(mask_L1_plus, -Y, P)

        # ------eval exact unconstrained step----------
        a=np.sum(grad_Y*P)
        b=np.sum((P.T@P)*H)
        # eta_star= -a/b if b>1e-14 else 1.0
        if b<1e-12:
            eta_star=1.0
        else:
            eta_star=-a/b
            minY=np.min(Y)
            boundary_thres=1e-5
            if minY>boundary_thres:
                lamda_max=1e20
            else:
                #  compute lambda_max when variables are near the bouddary
                lambda_max=1e20
            # if min_Z<= boundary_thres:
                mask_neg=P < 0.0
                if np.any(mask_neg):
                   ## Compute ratio only where D_t is negative; fill 1e20 everywhere else
                    ratio = np.where(mask_neg, -Y/P, lambda_max)
                    lambda_max = np.min(ratio)
            eta_ss=eta_star if eta_star < lamda_max else lamda_max
            if eta_ss<0.0:
                eta_ss=0.0

        # -----update ---
        B_new=Y+eta_ss*P   

        if variant=='AdaPGE1':
            tau_next=eta_star 
        else:
            grad_new=B_new@H+grad_const
            S=B_new-Y
            Y_diff=grad_new-grad_Y
            sy=np.sum(S*Y_diff)
            yy=np.sum(Y_diff*Y_diff)
            tau_next=sy/yy if yy>1e-12 else 1.0

        if variant=='AdaPGE3':
            alpha_min=tau_next/2.0
            alpha_max=2*tau_next

        alpha_cond=adap_switch(alpha_t,tau_next)
        alpha_next=max(alpha_min, min(alpha_max, alpha_cond))
        alpha_t=alpha_next
        B=B_new
        it=t+1
    return B, it
