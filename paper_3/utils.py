# -*- coding: utf-8 -*-
#
# Created on 2025-02-25 (Tuesday) at 14:48:00
#
# Author: Epsilon-79th
#
# Usage: Numerical experiments.
#

# Packages.
import time
import os, json

import numpy as np
import cupy as cp

from environment import *
import discretization as mfd
import dielectric as diel
from lobpcg import *
from pcfft import *

from numpy import pi

# Default package: cupy.
owari = owari_cuda
NP = cp

"""
Part 0: Initialization, recomputing and normalization, print.
"""

def uniform_initialization(n,d_flag,alpha,nev=NEV,k=K):
    
    """
    Usage:
        Trivial operations in matrix assembling.
    
    Input:
        n:       grid size.
        d_flag:  name of lattice type.
        alpha:   lattice vector.
    
    Output:
        a_fft:   fft3d blocks of A.
        b_fft:   fft3d blocks of B'B.
        inv_fft: fft3d blocks of inv(AA'+pnt B'B).
        x0:      initial guess.
        shift:   ensuring the insingularity of the system.
        
    """
    
    t_h = time.time()

    relax_opt, pnt = mfd.set_relaxation(alpha, scal=SCAL)

    ct = diel.diel_info(d_flag, option='ct')
    a_fft, b_fft = mfd.fft_blocks(n,k,ct,alpha = alpha,scal = SCAL)
    inv_fft = mfd.inverse_3_times_3_B(b_fft,pnt,relax_opt[0])

    a_fft /= SCAL
    b_fft = (pnt * b_fft[0] / SCAL / SCAL, pnt * b_fft[1] / SCAL / SCAL)
    inv_fft = (inv_fft[0] * SCAL * SCAL,inv_fft[1] * SCAL * SCAL)
    m = round(nev * relax_opt[1]) + nev

    x0 = cp.random.rand(3 * n ** 3,m) + 1j * cp.random.rand(3 * n ** 3,m)

    t_o = owari()
    print(f"Matrix blocks done, {t_o - t_h:<6.3f}s elapsed.")
    
    return a_fft, b_fft, inv_fft, x0, relax_opt[0]

def pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift = 0.0):
    
    """
    (Photonic Crystals / Mimetic Finite Difference / Matrix-free Handle.)
    Usage: 
        Matrix handle of A_func, H_func and P_func.
    """

    A_func = lambda x: AMA(x, a_fft, Diels) # AMA' handle.
    H_func = lambda x: AMA_BB(x, a_fft, b_fft, Diels, shift) # AMA+\gamma B'B handle.
    P_func = lambda x: H_block(x, inv_fft)  # Precondition inv(AA'+\gamma B'B) handle.
    
    return A_func, H_func, P_func

def recompute_normalize_print(lambdas_in, x, A_func, shift=0.0, scal = SCAL):
    """
    Usage:
        Recompute eigenvalues, normalize eigenvectors, print results.
        Output result table.
    Input:
        lambdas_pnt: Eigenvalues from lobpcg.
        x:           Eigenvectors from lobpcg.
        A_func:      Matrix handle of A.
        nev:         Number of desired eigenpairs.
        shift:       Shift of the system (default = 0).
        scal:        Scaling, lattice constant (default = 1).
    Output:
        lambdas_pnt: Recomputed eigenvalues.
        lambdas_re:  Recomputed and normalized eigenvalues.
    """
    
    t_h = time.time()
    adax = A_func(x)
    if shift > 0.0:
        lambdas_pnt = lambdas_in - shift
    else:
        lambdas_pnt = lambdas_in.copy()
    R = adax - x * lambdas_pnt
    lambdas_re = ((x.T.conj() @ adax).diagonal() / (x.T.conj() @ x).diagonal()).real

    # Check NaN (optional, usually unnecessary).
    nan_pnt = cp.where(cp.isnan(lambdas_pnt))[0]
    nan_re = cp.where(cp.isnan(lambdas_re))[0]

    # If pnt nan, re valid, yellow warning.
    # If both nan, red warning. 
    if len(nan_pnt) > 0:
        for ind in nan_pnt:
            if cp.isnan(lambdas_re[ind]):
                print(f"{RED}Warning: NaN occurs in both lambda_pnt and lambda_re, index = {ind}. "
                      f"Please run the program again.")
            else:
                print(f"{YELLOW}Warning: NaN occurs in lambda_pnt, index = {nan_pnt}, "
                      f"but same index in lambda_re is valid.")

    # If re nan, pnt valid, yellow warning and set re = pnt.       
    if len(nan_re) > 0:
        for ind in nan_re:
            if cp.isnan(lambdas_pnt[ind]) is False:
                lambdas_re[ind] = lambdas_pnt[ind]

    # Robust square roots (avoid very small negatives).
    def sqrt_robust(a):
        if (a <= 0) & (a > -1e-8):
            sqrt_a = 0.0
        else:
            sqrt_a = a**0.5
        return sqrt_a
        
    t_o = owari()
    flag_spurious = False
    
    print(f"Runtime for recomputing: {t_o - t_h:<6.3f}s.")
    print("| i  |    omega   |  omega_re  |  abs(omega - omega_re)  | residual  |")
    for i in range(len(lambdas_pnt)):
        l1 = sqrt_robust(lambdas_pnt[i]) * scal/(2*pi)
        l2 = sqrt_robust(lambdas_re[i]) * scal/(2*pi)
        print(f"| {i + 1:<2d} | {l1:<10.6f} | {l2:<10.6f} |        {abs(l1-l2):<10.3e}       | {norm(R[:,i]):<6.3e} |")
        lambdas_pnt[i], lambdas_re[i] = l1, l2
        if l1-l2>1e-3:
            flag_spurious = True
    
    if flag_spurious:
        raise ValueError(f"{RED}Spurious eigenvalues occur.{RESET}")
    
    return lambdas_pnt, lambdas_re

def condition_number(mat_in):
    """
    Compute the condition number of matrix.
    """

    if callable(mat_in):
        prec = lambda x:x
    else:
        d = cp.array(1.0/mat_in.diagonal())
        prec = lambda x: (x.T*d).T
    eig_s, iters = lobpcg_default(mat_in, prec = prec, nev = 2, rlx=4, info = True)
    print(f"Eig_s done, runtime = {iters[1]:<6.3f}s, {eig_s}.")

    eig_l, iters = lobpcg_default(mat_in, nev = 2, rlx=4, info = True, maxmin="max")
    print(f"Eig_l done, runtime = {iters[1]:<6.3f}s, {eig_l}.")
    print(f"Condition number of eps_loc: {eig_l[0]/eig_s[0]:<6.3f}.")

    return

def print_standard_deviation(lambdas_pnt, lambdas_re, nev=NEV):
    """
    Data postprocessing.
    """
    sd_pnt, sd_re = cp.std(lambdas_pnt,axis=0), cp.std(lambdas_re,axis=0)
    print("\nStandard deviation of each eigenvalue:")
    print("| i  |  std_pnt  |  std_re   |")
    for i in range(nev):
        print(f"| {i + 1:<2d} | {sd_pnt[i]:<6.3e} | {sd_re[i]:<6.3e} |")

def convergence_rate(residuals):
    def rated(x):   # Compute dampening rate by linear regression.  
        return np.polyfit(np.arange(len(x)), x, 1)[0]

    m0 = rated(np.log(residuals))
    print(f"\nGlobal average convergence rate: {np.exp(m0):<6.3f}.")

    n_half = len(residuals) // 2
    m1 = rated(np.log(residuals[:n_half]))
    m2 = rated(np.log(residuals[n_half:]))
    print(f"First half average convergence rate: {np.exp(m1):<6.3f}.")
    print(f"Second half average convergence rate: {np.exp(m2):<6.3f}.")

    return



def main():
    
    return
    
    
if __name__ == '__main__':
    main()