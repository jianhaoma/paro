# PARO: Piecewise‑Affine Regularization Optimization

This repository provides reference implementations for solving problems of the form

$$\min_{x\in\mathbb{R}^d} f(x) + \lambda R(x),$$

where `f` is a loss function and `R` is a **piecewise‑affine regularizer (PAR)** with efficient proximal operators and built‑in quantization utilities.

---

## Installation

```bash
python>=3.9
pip install numpy matplotlib
```
The code is pure Python; no build step required.

---

## Core abstractions

### `loss.py`
Abstract interface for smooth losses:
- `func(x) -> float`: objective value \(f(x)\)
- `grad_x() -> np.ndarray`: gradient at the **last** `x` used in `func`
- `grad(x) -> np.ndarray`: gradient at `x`
- `stochastic_grad(x) -> np.ndarray`: optional stochastic gradient
- `func_grad(x) -> (float, np.ndarray)`: value and gradient at `x`
- `global_min_x(x, rho) -> np.ndarray`: argmin of a quadratic-regularized model
  

Concrete implementations:
- `squared_loss.py: SquaredLoss(A, b)` – least‑squares \(\tfrac1{2n}\lVert Ax-b\rVert^2\)
- `logistic_loss.py: LogisticLoss(A, b)` – binary logistic NLL

Both losses additionally expose:
- `total_mvCount()`, `reset_mvCount()` – matvec accounting
- `initial_Lipschitz() -> float` – a quick L0 heuristic (power‑method style proxy)


### `regularizer.py`
Abstract interface for regularizers:
- `func(x) -> float`: value \(R(x)\)
- `prox_mapping(z, lambda_) -> x^+`: \(\arg\min_x \tfrac12\lVert x-z\rVert^2 + \lambda R(x)\)
- `scaled_prox_mapping(b, z, g, lambda_) -> x^+`: \(\arg\min_x \sum_i \tfrac12 b_i (x_i-z_i)^2 + g_i x_i + \lambda R(x)\)


Concrete implementation:
- `par.py: PAR(slopes, qs)` – piecewise‑affine regularizer parameterized by breakpoints `qs` and segment `slopes`.

  Key methods:
  - `func(x)` – evaluates \(R(x)\)
  - `prox_mapping(z, lambda_)`
  - `scaled_prox_mapping(b, z, g, lambda_)`
  - `is_convex()` – checks convexity from `slopes`
  - `calculate_quantization_rate(x, rtol=..., atol=...)`
  - `get_quantized_solution(x)`
  - (plot helpers may be present for quick visualization)

> The PAR class covers convex, certain nonconvex patterns, and a special “±1 alternating slopes” case with a closed‑form prox for grid‑like quantization.

---

## Solvers

### `prox_grad.py`
**Proximal Gradient (PG)** with line search:
```python
from prox_grad import prox_grad
logger = prox_grad(f, R, lambda_, x0, L0, tol)
```
- Backtracking line search using options from `algo_options.py`.
- Returns a `SolutionLogger` with histories and final solution.

### `acc_prox_grad.py`
**Accelerated PG**:

```python
from acc_prox_grad import acc_prox_grad
logger = acc_prox_grad(f, R, lambda_, x0, L0, tol)
```

### `ADMM.py`
**ADMM** with adaptive $\rho$ and primal/dual stopping criterion:

```python
from ADMM import ADMM
logger = ADMM(f, R, lambda_, x0, L0, tol)
```

All solvers read common flags from `algo_options.parse_args()`.

---

## Utilities

### `solution_logger.py`
`SolutionLogger` collects per‑iteration telemetry for plotting and diagnosis.

Fields (most relevant):
- `status`: `'U'` unknown, `'O'` optimal, `'M'` max‑iters
- `x`: final solution, `M`: local Lipschitz at termination
- `fx_all`, `Rx_all`, `Fx_all`: \(f(x), R(x), f(x)+\lambda R(x)\)
- `Lf_all`: local Lipschitz used by line search
- `NNZ_all`: nonzeros, `res_all`: optimality residuals (if computed)
- `nAx_all`: matvec counts
- `quantization_rate_all`: via PAR

Helpers:
- `record(...)`, `update_solution(...)`, `concatenate(next_logger)`

### `rand_model.py`
Synthetic data generator for quick tests:
```python
from rand_model import rand_model
A, b, x_true = rand_model(n=200, d=50, disAxz='nnn', Ascale=1.0, xscale=1.0, zscale=1.0, seed=0)
```
- `disAxz` chooses distributions for `(A, x, z)` using three letters (**n**ormal / **i**nteger / **s**parse‑integer / **z**eros), e.g., `'nnz'`.
- If `--loss logistic`, then `b` is drawn Bernoulli from `sigmoid(Ax+z)`; else `b = Ax + z`.


### `algo_options.py`
Centralized runtime flags (via `argparse`):
- General: `--verbose`, `--recording` (on by default)
- Problem: `--loss {l2, logistic}`, `--regularizer {convex, ...}`
- Algorithms:
  - `--line-search / --no-line-search`
  - `--stochastic / --no-stochastic`
  - `--stopbyKKT` (enable KKT‑based stopping when supported)
  - `--tolerance FLOAT` (default `1e-5`)
  - `--max_iters INT` (default `5000`)
  - Line‑search tuning: `--gamma_inc FLOAT` (default `2.0`), `--gamma_dec FLOAT` (default `2.0`), `--Lipsc_min FLOAT` (default `0.0`), `--Lipsc_max FLOAT` (default `1e8`)

> The solvers read these options at runtime; keep `parse_args()` accessible on your paths.

---

## Demo: compare PG / acc‑PG / ADMM

`test.py` runs a small synthetic experiment (Lasso‑style) and plots the **objective gap vs. iterations** for the three solvers.

Run:
```bash
python test.py --loss l2 --regularizer convex --line-search --tolerance 1e-6
# or a logistic variant
python test.py --loss logistic --regularizer convex --line-search
```

What it does (at a high level):
1. Parses CLI flags with `parse_args()`.
2. Generates data with `rand_model(...)` based on `--loss`.
3. Builds a simple convex PAR via `PAR(slopes=..., qs=...)`.
4. Sets `L0` (tries a Hessian diagonal heuristic; falls back to 1 if unavailable).
5. Runs `ADMM`, `prox_grad`, and `acc_prox_grad` with the same `(f, R, lambda_, x0, L0, tol)`.
6. Plots the convergence curves (log‑scaled objective gap).

The script stores all iterate histories in the returned `SolutionLogger`s (`logger_admm`, `logger_pg`, `logger_acc_pg`).

---

## Quick start (minimal)

```python
import numpy as np
from rand_model import rand_model
from squared_loss import SquaredLoss
from par import PAR
from prox_grad import prox_grad

# Data & loss
A, b, x_true = rand_model(n=200, d=50, disAxz='nnn', seed=0)
f = SquaredLoss(A, b)

# A simple convex PAR example
qs = np.array([0, 1, 2, 3], dtype=float)
slopes = np.array([1, 2, 3, 4], dtype=float)
R = PAR(slopes=slopes, qs=qs)

# Solve
x0 = np.zeros((A.shape[1], 1))
L0 = 2 * max(f.diag_Hessian())
lambda_ = 0.1
tol = 1e-6

logger = prox_grad(f, R, lambda_, x0, L0, tol)
print("status:", logger.status)
print("final objective:", logger.Fx_all[-1] if logger.Fx_all else None)
print("quantization rate:", logger.quantization_rate_all[-1] if logger.quantization_rate_all else None)
```

For logistic regression, swap `SquaredLoss` with `LogisticLoss` and regenerate `b` via `rand_model(..., disAxz='nnz')` with `--loss logistic`.
