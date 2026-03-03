# -*- coding: utf-8 -*-
#
# Created on 2026-01-13 (Tuesday) at 18:24:26
#
# Author: Epsilon-79th
#
# Usage: NEP Hermitian.
#        Fixed point, Newton iteration.
#

import gc
import numpy as np
from numpy import pi
import cupy as cp
import cupyx as cpx

from functools import partial
import logging

from environment import *
import dielectric as diel
from pcfft import *
from utils import *
from lobpcg import *

"""
    Auxiliary internal functions for NEP iteration.
"""

def _update_diel(idx, eps):
    # Update dielectric function handle.

    def diel_func_handle(x_in):
        x = x_in.copy()
        x[idx] *= eps
        return x
    return diel_func_handle

def _rayleigh_ama(x_in, a_fft, idx, eps):
    # Compute Rayleigh quotient x^H A x / x^H x, A is discrete curl.

    x = x_in.copy()
    n = round((x_in.size // 3)**(1/3))
    atx = A_block_kernel(x, -a_fft.conj())
    matx = cp.zeros_like(atx)
    matx[idx] = atx[idx]
    temp = cp.cublas.dotc(atx, matx).real
    
    if isinstance(eps, list):
        n_eps = len(eps)
        res = []
        for i in range(n_eps):
            res.append(eps[i]*temp)
        return tuple(res)
    else:
        return eps*temp

def _newton_update_freq(mu, freq_pre, x, a_fft, idx, nep_func):
    # Update frequency with one step of Newton iteration.

    freq_fpi = sqrt_robust(mu)
    dot_mu = _rayleigh_ama(x, a_fft, idx, nep_func.derivative(freq_pre)) - 2*freq_pre
    if abs(freq_pre) > 1e-10:
        freq_new = freq_pre - (mu - freq_pre*freq_pre) / dot_mu
        if nep_func(freq_new) < 0:
            return freq_fpi
        else:
            return freq_new
    else:
        return freq_fpi

"""
    Nonlinear iteration procedures.
"""

def fpi4nep(size, update_matrix, solver, nev, 
            relaxtion = None, tol = TOL, 
            newton_update = None, history = False):
    
    """
    Usage:
        Use fixed-point (FP) or Newton iteration to solve NEPs.
    
    Input:
        size(int):                  size of the problem.
        update_matrix(func handle): frequency(float) --> matrix handle.
        solver(func handle):        linear eigenproblem solver, 
                                    (h_func_in, x0, nev) --> (lambdas_pnt, x, iters).
        nev(int):                   number of eigenvalues to compute.
        relaxtion(func handle):     optional, iteration number --> subspace dimension.
        tol(float):                 optional, tolerance for convergence.
        newton_update(func handle): optional, if None then FP else Newton iteration.
        history(bool):              optional, whether to record convergence.
    
    Output:
        freqs_nrmlzd(np.ndarray):   computed frequencies normalized by 2*pi.
        eigvec(cp.ndarray):         computed eigenvectors.
        info(np.ndarray):           iteration number and runtime for each eigenvalue.
        history_nrm(np.ndarray):    optional, history of residual norms during iteration.
    
    """

    # Initialization.
    if relaxtion is None:
        relaxtion = lambda n: max(n+4, round(1.5*n))

    # History record.
    history_nrm = np.zeros((10000,))
    idx_his = 0

    freqs_nrmlzd = np.zeros(nev)
    eigvec = cp.empty((size, nev), dtype=complex)
    info = np.zeros((nev, 2))
    freq_pre = 0.0

    for i in range(nev):
        runtime = 0.0   # Runtime for current eigenvalue.

        t_h = time.time()
        m = relaxtion(i+1)
        if i < 1:   
            x0 = cp.random.rand(size,m) + 1j * cp.random.rand(size,m)
        else:
            # Extend initial guess.
            m0 = m - relaxtion(i)
            x0_extra = cp.random.rand(size,m0) + 1j*cp.random.rand(size,m0)
            x0_extra -= cp.cublas.gemm('N','N', x0, cp.cublas.gemm('H','N', x0, x0_extra))
            x0_extra /= cp.linalg.norm(x0_extra, axis=0)
            x0 = cp.hstack((x0, x0_extra))
        matrix_handle = update_matrix(freq_pre)
        iter_fp = 0
        step_length = 10.0 # preset.
        runtime += owari() - t_h    # Init runtime.

        while True:
            lambdas_pnt, x, iters = solver(h_func_in=matrix_handle, x0=x0, nev=i+1)
            runtime += iters[1]
            while x is None:
                # If solver fails, try random initial guess until success.
                # The failure usually occurs on the first few iterations since x0 from previous iteration
                # is "too close" to the current solution, causing NaN in Rayleigh-Ritz procedure.
                cp.cuda.Device().synchronize()
                print(f"{RED}Error occurs (nan or stagnation). Try random initial guess.{RESET}")
                gc.collect() 
                cp.get_default_memory_pool().free_all_blocks()
                x0 = cp.random.rand(size,m) + 1j*cp.random.rand(size,m)
                lambdas_pnt, x, iters = solver(h_func_in=matrix_handle, x0=x0, nev=i+1)
                runtime += iters[1]
                
            if step_length >= 1.0 or newton_update is None:
                freq_new = sqrt_robust(lambdas_pnt[i])
            else:
                freq_new = newton_update(lambdas_pnt[i], freq_pre, x[:,i])

            matrix_handle = update_matrix(freq_new)
            res_nrm = cp.linalg.norm(matrix_handle(x[:,i]) - freq_new*freq_new * x[:,i])
            history_nrm[idx_his+iter_fp] = res_nrm
            iter_fp += 1
            step_length = cp.abs(freq_new - freq_pre)

            # Optional relaxation.
            freq_new = freq_pre + 1.0*(freq_new - freq_pre)
            
            print(f"{CYAN}(i={i+1:3d}) Inner iter {iter_fp:2d}: residual = {res_nrm:<7.3e}, freq_nrmlzd = {freq_new/(2*pi):10.6f}, "
                  f"step of freq = {step_length:<7.3e}.{RESET}")

            freq_pre = freq_new
            gc.collect() 
            cp.get_default_memory_pool().free_all_blocks()
            cp.copyto(x0, x)
            if res_nrm < tol:
                break

        # Record results.
        freq_pre = sqrt_robust(lambdas_pnt[i+1])
        freqs_nrmlzd[i] = freq_new/(2*pi)
        idx_his += iter_fp
        cp.copyto(eigvec[:,i], x[:,i])
        info[i, 0], info[i, 1] = iter_fp, runtime
        print(f"\n{CYAN}i = {i+1:3d}-th eigenvalue ({freqs_nrmlzd[i]:12.6f}) converged, iter_fp = {iter_fp:3d}, runtime = {runtime:<6.2f}s.\n{RESET}")
    
    print(f"{CYAN}Total runtime = {np.sum(info[:,1]):.2f}s.{RESET}")
    if history:
        return freqs_nrmlzd, eigvec, info, history_nrm[:idx_his]
    else:
        return freqs_nrmlzd, eigvec, info

def nep_1p(n = 120, d_flag = SC_F1, no = 20, nev = NEV, problem_type = "QUAD", newton = True, tol = None):

    """
    Usage:
        Compute a single point in the band structure, with fixed-point iteration or Newton iteration.
        One may save convergence history in a json file under "output/" by specifying a tolerance `tol`.

    Input:
        n(int):                    grid size along each axis.
        d_flag(str):               lattice type, e.g. "SC_F1", "FCC", etc.
        no(int or np.ndarray):     index of the point in the Brillouin zone, or numpy.ndarray of shape (3,).
        nev(int):                  number of desired eigenvalues.
        problem_type(str):         "QUAD" for quadratic NEP, "SiC" for SiC NEP (see environment).
        newton(bool):              whether to use Newton iteration, if False then fixed-point iteration is used.
        tol(float):                optional, if specified then convergence history will be recorded and saved.
    
    Output:
        None.
    """

    if isinstance(no, int):
        alpha_sample = diel.diel_alpha(d_flag, no)
    else:
        # no is a numpy.ndarray
        alpha_sample = no.copy()
    a_fft, b_fft, inv_fft, _ , shift = uniform_initialization(n, d_flag, alpha_sample)
    p_func = partial(H_block, DIAG=inv_fft)
    idx = diel.diel_io_index(n, d_flag)
    x = cp.empty((3*n*n*n, nev), dtype=complex)

    nep_func = eval("NEP_EPS_"+problem_type)()
    def _update(freq):
        # Fix D_A, D_B, diel.
        return partial(AMA_BB, D_A=a_fft, D_B=b_fft, diel=_update_diel(idx, nep_func(freq)))
    
    # Define update strategy.
    if newton:
        newton_update = partial(_newton_update_freq, a_fft=a_fft, idx=idx, nep_func=nep_func)
        output_kwd = "newton"
    else:
        newton_update = None
        output_kwd = "fpi"
    
    # Fix precondition and shift.
    if tol is None:
        solver = partial(lobpcg_sep_softlock, p_func=p_func, shift=shift)
        freqs_nrmlzd, x, info = fpi4nep(3*n*n*n, _update, solver, nev, newton_update=newton_update)
    else:
        # Record history.
        solver = partial(lobpcg_sep_softlock, p_func=p_func, tol = tol*0.6, shift=shift)
        freqs_nrmlzd, x, info, history_nrm = fpi4nep(3*n*n*n, _update, solver, nev, tol=tol, newton_update=newton_update, history=True)
        path = OUTPUT_PATH+"history_"+output_kwd+"_"+d_flag+"_NEP_EPS_"+problem_type+".json"
        with open(path,'w') as file:
            json.dump({'res':history_nrm.tolist(), 'idx':info[:,0].astype(int).tolist()}, file, indent=4)

    print(f"{GREEN}\n{output_kwd} iteration for NEP {problem_type} completed, runtime = {np.sum(info[:,1]):.2f}s.{RESET}")
    # Investigate derivatives at computed frequencies.
    for i in range(nev):
        dot_mu = _rayleigh_ama(x[:,i], a_fft, idx, nep_func.derivative(freqs_nrmlzd[i]*2*pi))
        print(f"i = {i+1:2d}, freq = {freqs_nrmlzd[i]:12.6f}, fp iter = {int(info[i,0]):3d}, runtime = {info[i,1]:6.2f}s, "
              f"{CYAN}dmu_i = {dot_mu:<8.2e}, |fpi_rate| = {np.abs(dot_mu/(2*pi*freqs_nrmlzd[i])):8.3f}.{RESET}")

    return

def bandgap_nep(n=120, d_flag=SC_F1, problem_type="SiC", indices=None, newton = True):

    """
    Usage:
        Compute the bandgap along the high-symmetry points in the Brillouin zone, with fixed-point iteration or Newton iteration.
        The results will be saved in a json file under "output/" for future reference. 
        If a previous record exists, the program will automatically compute those uncomputed or failed points.
    
    Input:
        n(int):                    grid size along each axis.
        d_flag(str):               lattice type, e.g. "SC_F1", "FCC", etc.
        problem_type(str):         "QUAD" for quadratic NEP, "SiC" for SiC NEP (see environment).
        indices(list of int):      optional, indices of the points in the Brillouin zone to compute, if None then all points will be computed.
        newton(bool):              whether to use Newton iteration, if False then fixed-point iteration is used.
    
    Output:
        err_index(list of int):    indices of the points where error occurs (e.g. convergence failure).
    """

    # Fixed.
    gap = GAP
    nn = n * n * n

    # Load diel_info.
    ct, sym_points = diel.diel_info(d_flag)
    n_pt = sym_points.shape[0] - 1
    
    d_fft, di_fft = mfd.fft_blocks(n, K, ct)
    idx = diel.diel_io_index(n, d_flag)
    nep_class = eval("NEP_EPS_"+problem_type)
    nep_func = nep_class()

    # Define update strategy.
    output_kwd = "newton" if newton else "fpi"

    # Check history.
    path_bandgap = OUTPUT_PATH+"bandgap_"+output_kwd+"_"+d_flag+'_NEP_EPS_'+problem_type+".json"
    var_name_it = d_flag + "_" + str(n) + "_iters"
    var_name_fq = d_flag + "_" + str(n) + "_freqs"

    # File Check.
    uncomputed_ind = None    # If previous record exists but few indices remain uncomputed.
    if not os.path.exists(path_bandgap):
        # New lattice bandgap plot.
        
        print(f"The NEP {problem_type} has no previous record.")
        gap_rec_it, gap_rec_fq = [0] * (n_pt*gap), [[0] * NEV] * (n_pt * gap)
        gap_lib = {var_name_it: gap_rec_it, var_name_fq: gap_rec_fq}
        with open(path_bandgap,'w') as file:
            json.dump(gap_lib, file, indent = 4)
    else:
        with open(path_bandgap,'r') as file:
            gap_lib = json.load(file)
        if var_name_it in gap_lib.keys():
            # Previous record exists.
            print(f"{GREEN}The {output_kwd} - solved {problem_type} with grid size n = {n} has a previous record.{RESET}")
            gap_rec_it, gap_rec_fq = gap_lib[var_name_it], gap_lib[var_name_fq]
            
            # Get error, uncomputed index.
            err_ind = [i for i,a in enumerate(gap_rec_it) if a ==-1]
            if len(err_ind) > 0:
                print(f"{RED}Warning: Blow up results detected: {err_ind}.{RESET}")
                
            empty_ind = [i for i,a in enumerate(gap_rec_it) if a==0]
            if len(empty_ind) > 0:
                print(f"{YELLOW}Following indices remain uncomputed: {empty_ind}.{RESET}")
        
            if len(empty_ind) == 0 and len(err_ind) == 0:
                print(f"{GREEN}All indices of the NEP {problem_type}  have been computed without errors.{RESET}")
                return []
            uncomputed_ind = sorted(set(empty_ind + err_ind))
            del err_ind, empty_ind
        else:
            # New grid size.
            
            print(f"{YELLOW}The NEP {problem_type} will be computed with a new grid size n = {n}.{RESET}")
            
            gap_rec_it, gap_rec_fq = [0] * (n_pt * gap), [[0] * NEV] * (n_pt*gap)
            gap_lib[var_name_it], gap_lib[var_name_fq] = gap_rec_it, gap_rec_fq
            
            with open(path_bandgap,'w') as file:
                json.dump(gap_lib, file, indent = 4)

    if indices is None:
        indices = list(range(n_pt*gap)) if uncomputed_ind is None else uncomputed_ind
    
    """
    Calculation of bandgap.
    """
    
    err_index = []
    pool = cp.get_default_memory_pool()

    # Main Loop: compute each lattice point.    
    for i in range(len(indices)):

        t_h = time.time()
        
        alpha = diel.diel_alpha(d_flag, indices[i], gap)
        relax_opt, pnt = mfd.set_relaxation(alpha)
        
        a_fft = cp.concatenate((d_fft[:nn]+1j*alpha[0]*di_fft[:nn],\
                           d_fft[nn:2*nn]+1j*alpha[1]*di_fft[nn:2*nn],\
                           d_fft[2*nn:]+1j*alpha[2]*di_fft[2*nn:] ))
        b_fft = (cp.concatenate(((a_fft[0:nn]*a_fft[0:nn].conj()).real,\
                            (a_fft[nn:2*nn]*a_fft[nn:2*nn].conj()).real,\
                            (a_fft[2*nn:]*a_fft[2*nn:].conj()).real)),\
                 cp.concatenate((a_fft[0:nn].conj()*a_fft[nn:2*nn],\
                            a_fft[0:nn].conj()*a_fft[2*nn:],a_fft[nn:2*nn].conj()*a_fft[2*nn:])))
        inv_fft = mfd.inverse_3_times_3_B(b_fft,pnt,relax_opt[0])
        
        b_fft = (pnt*b_fft[0],pnt*b_fft[1])
        p_func = partial(H_block, DIAG=inv_fft)

        def _update(freq):
            # Fix D_A, D_B, diel.
            return partial(AMA_BB, D_A=a_fft, D_B=b_fft, diel=_update_diel(idx, nep_func(freq)))
        
        if newton:
            newton_update = partial(_newton_update_freq, a_fft=a_fft, idx=idx, nep_func=nep_func)
        else:
            newton_update = None
        
        print(f"Matrix blocks done, {owari()-t_h:<6.3f}s elapsed.")
        pool.free_all_blocks()
        freqs_nrmlzd, _ , info = fpi4nep(3*nn, _update, partial(lobpcg_sep_softlock, p_func=p_func), NEV, newton_update=newton_update)
        for j in range(NEV):
            print(f"i = {j+1:2d}, lambda = {freqs_nrmlzd[j]:12.6f}, {output_kwd} iter = {int(info[j,0]):3d}, runtime = {info[j,1]:6.2f}s.")
        
        gap_rec_it[indices[i]], gap_rec_fq[indices[i]] = np.sum(info[:,1]), freqs_nrmlzd.tolist()
        gap_lib[var_name_it], gap_lib[var_name_fq] = gap_rec_it, gap_rec_fq

        with open(path_bandgap,'w') as file:
            json.dump(gap_lib, file, indent=4)

        del a_fft, b_fft, inv_fft, p_func, _update
        pool.free_all_blocks()
            
        print(f"{GREEN}[{output_kwd}{str(n)}] Gap info library ({d_flag}:{problem_type}) is updated ({indices[i]}/{n_pt*gap}).{RESET}")
    
    if len(err_index) > 0:
        print(f"{RED}Error occurs to following indices:{RESET}")
        print(err_index)
    else:
        print(f"{GREEN}All indices computed correctly.{RESET}")
    
    return err_index

def precision_test(n=120, d_flag=SC_F1, problem_type="NEP_EPS_SiC", no=20, nev=10, tols = [1e-5, 1e-6, 1e-7]):

    """
    Usage:
        Test the effect of different tolerances on the computed frequencies.
    Input:
        n(int):                    grid size along each axis.
        d_flag(str):               lattice type, e.g. "SC_F1", "FCC", etc.
        problem_type(str):         "QUAD" for quadratic NEP, "SiC" for SiC NEP (see environment).
        no(int or np.ndarray):     index of the point in the Brillouin zone, or numpy.ndarray of shape (3,).
        nev(int):                  number of desired eigenvalues.
        tols(list of float):       list of tolerances to test.

    Output:
        None.
    """

    n_tol = len(tols)
    freqs_all = np.zeros((nev, n_tol))
    info = np.zeros((nev, 2, n_tol))

    # Initialization.
    if isinstance(no, int):
        alpha_sample = diel.diel_alpha(d_flag, no)
    else:
        # no is a numpy.ndarray
        alpha_sample = no.copy()
    a_fft, b_fft, inv_fft, _ , shift = uniform_initialization(n, d_flag, alpha_sample)
    p_func = partial(H_block, DIAG=inv_fft)
    idx = diel.diel_io_index(n, d_flag)

    nep_func = eval(problem_type)()
    def _update(freq):
        # Fix D_A, D_B, diel.
        return partial(AMA_BB, D_A=a_fft, D_B=b_fft, diel=_update_diel(idx, nep_func(freq)))
    
    for i in range(n_tol):
        solver = partial(lobpcg_sep_softlock, p_func=p_func, shift=shift, tol=tols[i])
        freqs_all[:,i], _ , info[:,:,i] = fpi4nep(3*n*n*n, _update, solver, nev, tol=tols[i])
        print(f"{GREEN}tol = {tols[i]:.2e} finished computing.\n{RESET}")
        for j in range(nev):
            print(f"{CYAN}i = {j+1:3d}, freq_nrmlz = {freqs_all[j,i]:12.6f}.{RESET}")

    for i in range(nev):
        print(f"i = {i+1:3d}, ", end = "")
        for j in range(n_tol):
            print(f"[tol {tols[i]:.2e}] ({freqs_all[i,j]:12.6f}, {int(info[i,0,j]):3d}, {info[i,1,j]:6.2f}s.", end="\t")
    return

def read_band_runtime(n, d_flag, problem_type):

    """
    Usage:
        Output the runtime record of bandgap calculation for a completely calculated case.

    Input:
        n(int), d_flag(str), problem_type(str): specify the case to check, e.g. n=120, d_flag="SC_F1", problem_type="SiC".
    """

    def _read_runtime(method):
        path_bandgap = OUTPUT_PATH + "bandgap_"+method+"_"+d_flag+'_NEP_EPS_'+problem_type+".json"
        with open(path_bandgap,'r') as file:
            gap_lib = json.load(file)
        var_name_it = d_flag + "_" + str(n) + "_iters"
        if var_name_it in gap_lib.keys():
            gap_rec_it = gap_lib[var_name_it]
            print(f"{GREEN}The {method} - solved {problem_type} with grid size n = {n} has a previous record.{RESET}")
            print(f"Average runtime = {np.mean(gap_rec_it):.2f}s, max runtime = {np.max(gap_rec_it):.2f}s, min runtime = {np.min(gap_rec_it):.2f}s.")

    _read_runtime("newton")
    _read_runtime("fpi")
    return

def main():
    
    #bandgap_nep(n=120, d_flag=SC_F1, problem_type="SiC", newton=True)
    #bandgap_nep(n=100, d_flag=SC_F1, problem_type="SiC", newton=False)
    #read_band_runtime(n=120, d_flag=SC_F1, problem_type="SiC")

    nep_1p(n=120, d_flag=FCC, no=20, nev=10, problem_type="SiC", newton=True)

    return

if __name__ == "__main__":
    main()
