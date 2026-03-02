# -*- coding: utf-8 -*-
#
# Created on 2025-09-27 (Saturday) at 16:31:50
#
# Author: Epsilon-79th.
#
# Usage: Global environment for computing photonic crystals.
#

import numpy as np
import cupy as cp

from numpy import pi
import time
from contextlib import contextmanager
from functools import wraps

# Output path.
OUTPUT_PATH = "output/"
DIEL_PATH = "dielectric_examples/"

# Global parameters.
K = 1                         # Stencil length.
NEV = 10                      # Number of desired eigenpairs.
SCAL = 1                      # Lattice scaling constant.
TOL = 1e-5                    # Tolerance.
GAP = 20                      # Segmentation of the Brillouin zone.

# Chiral constants (positive real).
CHIRAL_EPS_EG = {"sc_curv":13.0, "bcc_sg":16.0, "bcc_dg":16.0, "fcc":13.0}

# Pseudochiral constants (Hermitian postive definite, 3*3). 
PSEUDOCHIRAL_EPS_LOC = [np.array([(1+0.875**2)**0.5, (1+0.875**2)**0.5, 1.0, -1j*0.875, 0.0, 0.0]),\
                        np.array([(1+0.875**2)**0.5, 1.0, (1+0.875**2)**0.5, 0.0, 1j*0.875, 0.0]),\
                        np.array([1.0346,0.5059,0.2595, -0.0163-0.2319j, 0.027+0.0827j, -0.2743-0.0076j]),\
                        np.array([3.0, 3.0, 3.0, np.sqrt(3)+1j, 1j, np.sqrt(2)*(1+1j)])/5.0]

# Linear/Eigensolver settings.
MAXITER = 500
RESTART_MAX = 100
N_SUBSPACE = 40

# Lattice type.
SC_F1  = 'sc_flat1'
SC_F2  = 'sc_flat2'
SC_C   = 'sc_curv'
BCC_SG = 'bcc_sg'
BCC_DG = 'bcc_dg'
FCC    = 'fcc'

# Robust square root.
sqrt_robust = lambda x: 0 if x<1e-10 else x**0.5

# Omega_c: upper bound of frequency.
Omega_c = lambda w0, w1: (w1*w1-(w1*w1*(w1*w1-w0*w0))**0.5)**0.5

# Nonlinear dielectric function.
# The input frequency is \omega instead of \omega/2\pi.
class NEP_EPS_QUAD:
    # eps(x) = eps_inf * (w_L**2-x**2) / (w_T**2-x**2).
    # Return inverse.

    def __init__(self, eps_inf=20.0, w_T=8.12, w_L=8.75):
        self.eps_inf = eps_inf
        self.lamT = w_T**2
        self.lamL = w_L**2

    def value(self, x):
        eps = self.eps_inf * (self.lamL - x**2) / (self.lamT-x**2)
        return 1/eps
    
    def derivative(self, x):
        # Derivative of the inverse.
        deps = 2*x*(self.lamT-self.lamL) / ((self.lamL-x**2)**2) / self.eps_inf
        return deps
    
    def __call__(self, x):
        return self.value(x)
    
class NEP_EPS_QUAR:

    def __init__(self, eps_1=5.8, eps_inf=18.6, w_T=9.89, w_L=10.45):
        self.eps_1 = eps_1
        self.eps_inf = eps_inf
        self.lamT = w_T**2
        self.lamL = w_L**2
    
    def value(self, x_):
        x = x_ 
        eps = self.eps_1 + self.eps_inf * \
              (self.lamL - 0.3*x**2 - 0.25*x**3 + 0.2275*x**4) / \
              (self.lamT - 1.3*x - 0.3*x**2 - 0.125*x**3)
        return 1/eps
    
    def derivative(self, x):

        N = self.lamL - 0.3*x**2 - 0.25*x**3 + 0.2275*x**4
        D = self.lamT - 1.3*x - 0.3*x**2 - 0.125*x**3
        val = self.eps_1 + self.eps_inf * N / D

        N_prime = -0.6*x - 0.75*x**2 + 0.91*x**3
        D_prime = -1.3 - 0.6*x - 0.375*x**2
    
        return -(self.eps_inf * (N_prime*D - N*D_prime) / (D**2)) / (val**2)

    def __call__(self, x):
        return self.value(x)
    
class NEP_EPS_SiC():

    def __init__(self, eps_inf=6.7, w_T=0.817, w_L=0.998):
        self.eps_inf = eps_inf
        self.lamT = (w_T*2*pi)**2
        self.lamL = (w_L*2*pi)**2
    
    def value(self, x):
        eps = self.eps_inf * (self.lamL - x**2) / (self.lamT - x**2)
        return 1/eps
    
    def derivative(self, x):
        deps = 2*x*(self.lamT - self.lamL) / ((self.lamL - x**2)**2) / self.eps_inf
        return deps
    
    def __call__(self, x):
        return self.value(x)
    
class NEP_EPS_EXP():

    def __init__(self, eps_inf=11.4, beta = 0.8, tau=0.2):
        self.eps_inf = eps_inf
        self.beta = beta
        self.tau = tau
    
    def value(self, x):
        eps = self.eps_inf - self.beta * np.exp(-x*self.tau) 
        return 1/eps
    
    def derivative(self, x):
        deps = self.beta * self.tau * np.exp(-x*self.tau)
        return -deps * (self.value(x)**2)
    
    def __call__(self, x):
        return self.value(x)

class NEP_EPS_MIX():

    def __init__(self, eps_1=11.8, lamT=22.8, lamL=6.7):
        self.eps_1 = eps_1
        self.lamT = lamT
        self.lamL = lamL

    def value(self, x):
        eps = self.eps_1 + (self.lamL - 1.2*x+2*x**2+0.11*x**3) / (self.lamT - 0.3*x - 0.025*x**2) - 1.4 * np.exp(-1.08*x)
        return 1/eps
    
    def derivative(self, x):
        N  = 22.8 - 1.2*x + 2.0*x**2 + 0.11*x**3          # N(x)
        N1 = -1.2 + 4.0*x + 0.33*x**2                     # N'(x)
    
        D  = 6.7 - 0.3*x - 0.025*x**2                     # D(x)
        D1 = -0.3 - 0.05*x                                # D'(x)
    
        frac_deriv = (N1 * D - N * D1) / (D * D)
        exp_deriv = 1.512 * np.exp(-1.08 * x)
    
        return -(frac_deriv + exp_deriv) * (self.value(x)**2)
    
    def __call__(self, x):
        return self.value(x)


# Color.
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"
RESET   = "\033[0m"

# Dielectric domain, symmetric points.
DIEL_LIB={'CT_sc':[[1,0,0],[0,1,0],[0,0,1]],\
          'CT_bcc':[[0,1,1],[1,0,1],[1,1,0]],\
          'CT_fcc':[[-1,1,1],[1,-1,1],[1,1,-1]],\
          'sym_sc':[[0,0,0],[pi,0,0],[pi,pi,0],\
                    [pi,pi,pi],[0,0,0]],\
          'sym_bcc':[[0,0,2*pi],[0,0,0],[pi,pi,pi],\
                     [0,0,2*pi],[pi,0,pi],[0,0,0],\
                     [0,2*pi,0],[pi,pi,pi],[pi,0,pi]],\
          'sym_fcc':[[0,2*pi,0],[pi/2,2*pi,pi/2],[pi,pi,pi],\
                     [0,0,0],[0,2*pi,0],[pi,2*pi,0],\
                     [3*pi/2,3*pi/2,0]]}

@contextmanager
def timing(process_name=None, time_array=None, index=None, runtime_dict=None, print_time=False, accumulate=False):
    """
    Usage:
        General TIMING contextmanager.
    Input:
        process_name: Name of process.
        time_array:   Storing time.
        index:        Index of time array.
        runtime_dict: Dict storing time.
        print_time:   Whether print time.
        accumulate:   Whether accumulate time.
    """
    
    t_h = time.time()
    yield
    t_o = owari_cuda()
    elapsed = t_o - t_h

    if time_array is not None and index is not None:
        if accumulate:
            time_array[index] += elapsed
        else:
            time_array[index] = elapsed
    if runtime_dict is not None and process_name is not None:
        runtime_dict[process_name] = runtime_dict.get(process_name, 0) + elapsed
    if print_time and process_name is not None:
        print(f"Runtime of {process_name} is {elapsed:<6.3f} s.")
    

"""
Norms.
"""

def norm(X):
    
    """
    Input:  matrix X.
    Output: Frobenius norm.
    """    

    NP=cp if isinstance(X,cp.ndarray) else np
    
    if X.ndim<=1:
        return NP.linalg.norm(X)
    else:
        return NP.sqrt((NP.trace(NP.dot(X.T.conj(),X))).real)

def norms(X):
    
    """
    Input:  multicolumn vector X.
    Output: an array containing norm of each column.
    """  
    
    NP=cp if isinstance(X,cp.ndarray) else np
    
    if X.ndim<=1:
        return NP.linalg.norm(X)
    else:
        return NP.sqrt((NP.diag(NP.dot(X.T.conj(),X))).real)
    
def dots(X,Y):
    
    """
    Input:  multicolumn vector X,Y (same column number).
    Output: an array containing inner product of each column of X,Y, diag(X^H*Y).
    """
    
    NP = arrtype(X)

    if X.ndim <= 1:
        return X.conj() @ Y
    else:
        return NP.diag(X.T.conj() @ Y)
    

"""
Package and GPU timing.
"""

def arrtype(x):
    
    if "numpy" in str(type(x)):
        return np
    elif "cupy" in str(type(x)):
        return cp
    else:
        ValueError("The input should either be a numpy.ndarray or cupy.ndarray.")
        
def owari_opt(T=None):
    
    return lambda: (cp.cuda.Device().synchronize(),time.time())[1]\
        if not T is None and (T=="cp" or T=="gpu" or "cupy" in T)\
        else time.time()

def owari_cuda():
    return (cp.cuda.Device().synchronize(),time.time())[1]