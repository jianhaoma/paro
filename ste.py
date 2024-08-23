import numpy as np
from algo_options import parse_args
from solution_logger import SolutionLogger
import matplotlib.pyplot as plt


def custom_round(array):
    # round it to the nearest integer
    # return np.sign(array) * np.floor(np.abs(array))  
    return np.round(array)


def ste(f, R, lam, x0, L0, tol):
    args = parse_args()

    # create logger to store solution history
    logger = SolutionLogger('STE', x0, L0)

    # initialization
    x = x0.copy()
    y = x0.copy()
    eta0 = 0.1 / L0
    eta_sum = 0
    M = L0
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
            Rx = R.func(x)
            logger.record(k=k, f=f, x=x, fx=fx, Rx=Rx, Fx=fx + lam * Rx, M=M, residue=residue, quantized_fx=fx)

        # proximal gradient descent update
        eta = eta0 / k
        y -= eta * gx
        x = custom_round(y)

        fx = f.func(x)

        '''TODO: add the check for stopping criterion'''
        # check for stopping criterion
        # if fx - fx1 <= tol:
        #     logger.status = 'O'
        #     break

    # maximum number of iterations reached
    if logger.status != 'O' and logger.status != 'L':
        logger.status = 'M'

    # record the latest solution vector and number of iterations
    logger.update_solution(x, M, lam, k)

    return logger
