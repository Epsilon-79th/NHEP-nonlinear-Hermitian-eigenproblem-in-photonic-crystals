# NHEP: Nonlinear Hermitian Eigenproblem in 3D Photonic Crystals

**Version:** [![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/) 

**Platform:** [![GPU](https://img.shields.io/badge/GPU-RTX_4090_24GB-76b900?style=flat&logo=nvidia)]() [![CUDA](https://img.shields.io/badge/CUDA-12.4-76b900?style=flat)]()

**Core Library 🚀:** [CuPy](https://cupy.dev/) for accelerated linear algebra.



## Overview

This is a **Python**-based computational mathematics project for 3D photonic crystals with frequency-dependent, non-loss dielectric coefficient function. It accompanies the paper *Kernel Compensation-Based Newton Iteration for Nonlinear Hermitian Eigenproblems in Photonic Crystals* (Chinese: 光子晶体中非线性Hermitian特征值问题的零空间补偿 Newton迭代法).

The repository is outline as follows:

```
├── paper_3/
│	├── pycache/
│	├── dielectric_examples/ 
│	│ 	└── edge_dofs/ 		 # Edge degree-of-freedom examples
│	├── output/ 			 # Output directory (bandgap, convergence history)
│	├── init.py 			 # Package initializer
│	├── _kernels.py 		 # CUDA kernels (cupy.ElementwiseKernel), imported by pcfft.py
│	├── dielectric.py 	     # I/O and computation of dielectric indices
│	├── discretization.py    # Finite difference, matrix-free blocks
│	├── environment.py       # Environment (parameters setting)
│	├── lobpcg.py            # Custom-implemented LOBPCG eigensolver
│	├── orthogonalization.py # Orthogonalization routines for LOBPCG
│	├── paper_3_test.py      # Main test script
│	├── pcfft.py             # GPU accelerated matrix-free operations via FFT
│	├── run.sh 				 # Shell run script (calls paper_3_test.py)
│	└── utils.py 		     # Initialization and postprocessing
├── pic/					 # Figures of material, bandgap and convergence history
├── README.md
└── paper_3_final.pdf
```



## Dependencies

- Python 3.10+
- `NumPy` 1.22+,  `SciPy` 1.7+,  `CuPy` 13.0+  (necessary for GPU acceleration)

The installation of above packages:

```
pip install numpy scipy cupy
```

You can also install the CUDA-specific wheel (e.g. cuda12x):

```
pip install cupy-cuda12x
```



## Running Examples

```bash
cd paper_3
```

To start the program, one may call shell file `run.sh`, which automatically detects available GPUs: the default threshold is set to be $100$ MiB, any GPU with occupied memory less than the threshold is considered to be available. 

```bash
./run.sh
```

One can also use direct command (in case of CUDA OOM, please make sure at least **16GiB** memory is available on your device):

```bash
python paper_3_test.py
```

If you know a specific GPU is free (e.g. the 4th GPU), use command

```bash
CUDA_VISIBLE_DEVICES=4 python paper_3_test.py   # if you know GPU 4 is available
```



## Test Functions of paper_3_test.py

We summarize the usage of test functions of `paper_3_test.py` in the following table. Detailed explanations of input and output are shown in the comments at the header docstring of the each function.

| Function Name                                                | Usage                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `_update_diel(idx, eps)`                                     | Update dielectric function handle: returns a function that multiplies the entries at indices `idx` by `eps`. |
| `_rayleigh_ama(x_in, a_fft, idx, eps)`                       | Compute the Rayleigh quotient \(x^H A x / x^H x\) for the discrete curl A. |
| `_newton_update_freq(mu, freq_pre, x, a_fft, idx, nep_func)` | Perform one Newton step to update the frequency based on the current residual and the derivative of the NEP. |
| `fpi4nep(size, update_matrix, solver, nev, relaxation=None, tol=TOL, newton_update=None, history=False)` | Core fixed‑point / Newton iteration for NEPs.                |
| `nep_1p(n=120, d_flag=SC_F1, no=20, nev=NEV, problem_type="QUAD", newton=True, tol=None)` | Compute a single point in the band structure using either fixed‑point or Newton iteration. Convergence history is saved to a JSON file if `tol` is specified. |
| `bandgap_nep(n=120, d_flag=SC_F1, problem_type="SiC", indices=None, newton=True)` | Compute the bandgap along high‑symmetry points in the Brillouin zone. Automatically resumes from previous runs and saves results to a JSON file. |
| `precision_test(n=120, d_flag=SC_F1, problem_type="NEP_EPS_SiC", no=20, nev=10, tols=[1e-5,1e-6,1e-7])` | Test the effect of different solver tolerances on the computed frequencies. Runs the NEP solver for each tolerance and compares the results. |
| `read_band_runtime(n, d_flag, problem_type)`                 | Read and print runtime from a completed bandgap calculation stored in a JSON file. |
| `main()`                                                     | Revise your test function to run under the main function.    |



## Results

The bandgap figures are stored in PDF format (which Github does NOT support) in the folder `pic/`. One can also find them in the paper. We present the runtime of each example in the table below (the physical parameters can be found in the paper, or our another repository https://github.com/Epsilon-79th/linear-eigenvalue-problems-in-photonic-crystals):

Average Runtime Comparison for Sample Problems Using Fixed-point Iteration and Newton Iteration (in seconds)

| Problem Type   | $3 \times 100^3$ DoFs | $3\times120^3$ DoFs   |
| -------------- | --------------------- | --------------------- |
|                | Newton Iteration      | Fixed-point Iteration |
| sc_flat1, SiC  | 119.16                | 183.39                |
| sc_flat2, QUAR | 97.20                 | 113.50                |
| sc_curv, MIX   | 60.86                 | 69.02                 |
| fcc, QUAR      | 216.86                | 458.54                |
