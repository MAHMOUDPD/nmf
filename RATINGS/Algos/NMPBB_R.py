
import numpy as np
import time
from numba import njit

# --- Helper functions (Compiled with Numba) ---
@njit(cache=True, fastmath=True)
def proj(X):
    return np.maximum(X, 0.0)

# @njit(cache=True, fastmath=True)
# def compute_obj(C, A, B, N, mu):
#     """
#     Compute objective (4):
#         f(B) = (1/(2N)) ||C - B A^T||_F^2 + (mu/2) ||B||_F^2
#     in the trace form
#         0.5 * (tr1 - 2 <B, VHT> + <B^T B, HHT>).
#     """
#     r    = A.shape[1]
#     tr1  = np.sum(C * C) / N
#     VHT  = (C@A) / N
#     HHT  = (A.T@A) / N + mu * np.eye(r)
#     return 0.5 * (tr1 - 2.0 * np.sum(B * VHT) + np.sum((B.T @ B) * HHT))
# def compute_obj_BA(C, A, B, N, mu):
#     """
#     Compute  f(B) = (1/(2N)) ||C - B A||_F^2 + (mu/2) ||B||_F^2
#     with C in R^{p x q},  B in R^{p x r},  A in R^{r x q}.
#     """
#     r    = B.shape[1]
#     tr1  = np.sum(C * C) / N
#     VHT  = (C @ A.T) / N                     # note: A.T, not A
#     HHT  = (A @ A.T) / N + mu * np.eye(r)    # note: A @ A.T, not A.T @ A
#     return 0.5 * (tr1 - 2.0 * np.sum(B * VHT) + np.sum((B.T @ B) * HHT))


@njit(cache=True, fastmath=True)
def solve_NMPBB_R(C, A, B0, N, mu, max_iter=200, tol=1e-7):
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

    AAt = A@A.T
    # AtX = A.T @ C
    tr1  = np.sum(C*C)/N

    H = AAt/N + mu*np.eye(r)                 # Hessian of the subproblem
    L=np.linalg.norm(H, 2)                     # Lipschitz constant of the gradient
    if L == 0.0 or np.isnan(L) or np.isinf(L):
        return B, 0.0
    CA=(C@A.T)/N
    grad_const= -CA    
                         # constant part of the gradient
    alpha_max = 1e20
    alpha_min = 1e-20
    alpha_t = 1.0
    rho = 0.25
    eta0=.1
    gamma = 1e-4
    mu1 = 0.5
    c = 1e-4
    s = 1.7 
    
    S_val = 0.0
    f_Z_prev = 0.0

    it=0
    for t in range(max_iter):
        # ---gradient ---
        grad = B@H + grad_const
        if not np.all(np.isfinite(grad)):
            print("Warning: Non-finite gradient encountered.")
            break
        # ----projected - gradient stopping criterion ---
        if np.linalg.norm(proj(B - grad) - B) < tol:
            break
        # --- proximal point step ---
        Y=proj(B - (1.0/L)*grad)
        # --- active set identificatiion ---
        grad_Y=Y@H + grad_const

        fB=0.5*(tr1 - 2.0 * np.sum(B*CA) + np.sum((B.T @ B)*H))
        if t==0:
            S_val=fB
        else:
            eta_t=(2.0/np.pi)*np.arctan(abs(fB-f_Z_prev))
            S_val=fB + eta_t*(S_val-fB)
        f_Z_prev=fB

        alpha_inv=1.0/alpha_t
        mask_active = (Y <= alpha_inv*grad_Y)
        mask_L1 = mask_active & (grad_Y>= c)
        mask_not_L1 = ~mask_L1
        # --- direction P^t---
        BB_step = proj(Y - alpha_t * grad_Y) - Y
        # 1. Assign BB_step to not L1, and 0 to L1
        Pt=np.where(mask_not_L1, BB_step, 0.0)
        # 2. overwrite Pt with -Y for L1
        mask_L1_plus = mask_L1 &(Y!=0.0)
        Pt=np.where(mask_L1_plus, -Y, Pt)
        # ------nonmonotone linesearch----------
        eta_t=1.0
        j=0
        while True:
            B_new=Y+s*eta_t*Pt
            fB_new=0.5*(tr1 - 2.0 * np.sum(B_new * CA) + np.sum((B_new.T @ B_new) * H))
            S_cond=fB_new+(0.5 if t==0 else eta_t)*(S_val-fB_new)
            Cond=np.sum(grad_Y*Pt)+(mu1/alpha_t)*np.sum(Pt*Pt)
            if S_cond <= S_val + gamma*eta_t*Cond:
                break
            eta_t*=rho
            j=j+1
            if eta_t < 1e-20:
                break
        it=it+j
        # -----update ---
        B_new=Y+s*eta_t*Pt   

        grad_new=B_new@H+grad_const
        Xt=B_new-Y
        Yt=grad_new-grad_Y
        xy=np.sum(Xt*Yt)
        xx=np.sum(Xt*Xt)
        tau_next=xx/xy if xy>1e-12 else 1.0

        alpha_next=max(alpha_min, min(alpha_max, tau_next))
        alpha_t=alpha_next
        B=B_new
        it=t+1
    return B, it
