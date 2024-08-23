import numpy as np
from algo_options import parse_args
from solution_logger import SolutionLogger
import matplotlib.pyplot as plt


def custom_round(x):
    # a = np.linalg.norm(y, ord=1) / n
    # return a * np.sign(x)
    return np.sign(x) * np.floor(np.abs(x))


def new_ste(f, R, lambda_, x0, L0, tol):
    args = parse_args()

    # create logger to store solution history
    logger = SolutionLogger('new STE', x0, L0)

    # initialization
    x = x0.copy()
    y = x0.copy()
    n = len(x0)
    eta0 = 0.1 / L0
    eta_sum = 0
    M = L0
    residue = 1e6

    fx = f.func(x)
    for k in range(1, args.max_iters + 1):
        # evaluate function and gradient at x
        # gx = f.grad(x)
        gx = f.stochastic_grad(x)

        # logger keeps record of solution path
        if args.recording:
            Rx = R.func(x)
            logger.record(k=k, f=f, x=x, fx=fx, Rx=Rx, Fx=fx + lambda_ * Rx, M=M, residue=residue, quantized_fx=fx)

        # gradient descent and quantization step
        eta = eta0 / np.sqrt(k)
        eta_sum += eta
        # y -= eta * 0.001 * y # weight decay
        y -= eta * gx
        x = custom_round(y * (5 / (5 + eta_sum)))

        fx = f.func(x)

        # check for stopping criterion
        # if fx - fx1 <= tol:
        #     logger.status = 'O'
        #     break

    # maximum number of iterations reached
    if logger.status != 'O' and logger.status != 'L':
        logger.status = 'M'
    # print('the magnitude is ', x[0])
    # record the latest solution vector and number of iterations
    logger.update_solution(x, M, lambda_, k)

    return logger
