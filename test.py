import numpy as np
from algo_options import parse_args
from solution_logger import SolutionLogger
import matplotlib.pyplot as plt
from par import PAR
from rand_model import rand_model
from ADMM import ADMM
from prox_grad import prox_grad
from acc_prox_grad import acc_prox_grad
from logistic_loss import LogisticLoss
from squared_loss import SquaredLoss

args = parse_args()


# Generate synthetic data
seed = np.random.seed(42)
n, d = 10, 100
A, b, x = rand_model(n=n, d=d, disAxz='nnn', seed=seed)

# construct objective functions
if args.loss == 'logistic':
    f = LogisticLoss(A, b)
elif args.loss == 'l2':
    f = SquaredLoss(A, b)

# Regularization parameter
lambd = 0.1 / np.sqrt(n)

# Convex PAR
k = 10
qs = np.arange(0, k, dtype=np.float64)
slopes = np.arange(1, k + 1, dtype=np.float64)

R = PAR(slopes=slopes, qs=qs)

try:
    L0 = 2 * max(f.diag_Hessian())
except:
    L0 = 1

x0 = np.zeros((d, 1))
logger_admm = ADMM(f, R, lambd, x0, L0, args.tolerance)
logger_pg = prox_grad(f, R, lambd, x0, L0, args.tolerance)
logger_acc_pg = acc_prox_grad(
    f, R, lambd, x0, L0, args.tolerance)

# solve the problem to higher accuracy to get the optimal objective value
logger = acc_prox_grad(f, R, lambd, logger_acc_pg.x,
                       L0, args.tolerance * 0.001)
Fx_min = min(logger.Fx_all)

# Plot the difference to the optimal objective value
plt.figure(figsize=(8, 6))
plt.plot((np.array(logger_admm.Fx_all) - Fx_min),
         label='admm', linewidth=2)
plt.plot((np.array(logger_pg.Fx_all) - Fx_min),
         label='pg', linewidth=2)
plt.plot((np.array(logger_acc_pg.Fx_all) - Fx_min),
         label='acc_pg', linewidth=2)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Difference to Optimal')
plt.title('Convergence of Different Algorithms for Lasso')
plt.legend()
plt.grid(True)
plt.show()
