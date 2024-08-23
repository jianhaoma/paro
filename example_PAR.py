import numpy as np
import datetime
import os
import time
from rand_model import rand_model
from PAR_by_choice import PAR_by_choice
from squared_loss import SquaredLoss
from logistic_loss import LogisticLoss
from par import CONVEX_PAR, QUASICONVEX_PAR, NONCONVEX_PAR
from prox_grad import prox_grad
from algo_options import parse_args
from plot_logs import plot_logs

# generate the data folder to store the results
current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y-%m-%d-%H:%M')
full_path = os.path.join("data", formatted_time)
if not os.path.exists(full_path):
    os.makedirs(full_path)


# generate a random instance
n = 10 # sample size
d = 100 # dimension
args = parse_args()
distr = 'nnn'  # distributions for generating A, x_bar, and z: normal for A, integer for x, zero for z.
# distr = 'nnz'
Ascale = 1
xscale = 2
zscale = 0.2
seed = 2024

A, b, x = rand_model(n, d, distr, Ascale, xscale, zscale, seed)

# construct objective functions
if args.loss == 'logistic':
    f = LogisticLoss(A, b)
elif args.loss == 'l2':
    f = SquaredLoss(A, b)

lam0 = np.linalg.norm(A.T.dot(b), ord=np.inf) / np.sqrt(d)

# generate a, q, b_ for the convex PAR
a = np.arange(1, args.k + 1, dtype=float)
q = np.arange(args.k)
b_ = np.zeros_like(a) # since b is already used to stand for the response vector
for i in range(1, args.k):
    b_[i] = b_[i - 1] + a[i - 1] * (q[i] - q[i - 1])

R_convex = CONVEX_PAR(a=a, b=b_, q=q)
R_quasiconvex = QUASICONVEX_PAR()
R_nonconvex = NONCONVEX_PAR()

# set th regularization parameter
lam = 0.05

algos = ['STE', 'PG'] # list of algorithms to run
loggers = [] # store the solution loggers

for algo in algos:
    start_time = time.time()
    logger = PAR_by_choice(A, b, f, R_convex, lam, algo)
    end_time = time.time()
    T = end_time - start_time
    print(f"\t{algo}\t{T:8.2f} sec")
    loggers.append(logger)

# visualization
print_eps = True
if args.recording:
    # approximate Fx_min by solving the problem to much higher precision
    if args.loss == 'logistic':
        f = LogisticLoss(A, b)
    elif args.loss == 'l2':
        f = SquaredLoss(A, b)
    logger = prox_grad(f, R_convex, lam, loggers[-1].x, loggers[-1].M, 1.0e-7)
    Fx_min = min(logger.Fx_all)
    # plot figures
    plot_logs(x, Fx_min, lam0, print_eps, full_path, *loggers)

# record hyperparameters into the log
with open(full_path + '/records.txt', 'w') as f:
    f.write(f'Loss: {args.loss}\n')
    f.write(f'Dimension: {d}\n')
    f.write(f'Sample size: {n}\n')
    f.write(f'Algorithms: {algos}\n')
    f.write(f'A scale: {Ascale}\n')
    f.write(f'Lambda: {lam}\n')
