import numpy as np
from algo_options import parse_args
from calculate_quantization_ratio import calculate_quantization_ratio
from solution_logger import SolutionLogger
import matplotlib.pyplot as plt


def proxquant(f, R, lam, x0, L0, tol):
    args = parse_args()

    # create logger to store solution history
    logger = SolutionLogger('PROXQUANT', x0, L0)

    # initialization
    x = x0.copy()
    y = x0.copy()
    eta0 = 0.5 / L0
    M = L0

    # for optimality test in the first iteration, use fake gradient mapping
    gMx = np.ones_like(x) * 1e10
    residue = 1e6

    fx = f.func(x)
    for k in range(1, args.max_iters + 1):
        # calculate the (stochastic) gradient at x
        if args.stochastic:
            gx = f.stochastic_grad(x)
        else:
            gx = f.grad_x()

        # logger keeps record of solution path
        if args.recording:
            quantization_ratio = calculate_quantization_ratio(x.reshape(-1))
            Rx = R.func(x)
            logger.record(k=k, f=f, x=x, fx=fx, Rx=Rx, Fx=fx + lam * Rx, M=M, residue=residue, quantization_ratio=quantization_ratio)

        # gradient descent step
        eta = eta0
        y -= eta * gx
        # proximal step
        x = R.prox_mapping(y, lam * eta * k)

    '''TODO: add the check for stopping criterion'''

    # maximum number of iterations reached
    if logger.status != 'O' and logger.status != 'L':
        logger.status = 'M'

    # record the latest solution vector and number of iterations
    logger.update_solution(x, M, lam, k)

    return logger
