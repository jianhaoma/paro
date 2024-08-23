import numpy as np
from algo_options import parse_args
from solution_logger import SolutionLogger
from calculate_quantization_ratio import calculate_quantization_ratio
import matplotlib.pyplot as plt

def parq(f, R, lambda_, x0, L0, tol):
    args = parse_args()

    # create logger to store solution history
    logger = SolutionLogger('PARQ', x0, L0)

    # initialization
    x = x0.copy()
    y = x0.copy()
    eta0 = 0.1 / L0
    gamma_inv = 0 # sum of etas
    M = L0

    # for optimality test in the first iteration, use fake gradient mapping
    gMx = np.ones_like(x) * 1e10
    residue = 1e6

    fx = f.func(x)
    print(args.max_iters)
    for k in range(1, args.max_iters + 1):
        # calculate the (stochastic) gradient at x
        if args.stochastic:
            gx = f.stochastic_grad(x)
        else:
            gx = f.grad_x()

        # logger keeps record of solution path
        if args.recording:
            quantized_x = np.round(x)
            quantized_fx = f.func(quantized_x)
            Rx = R.func(x)
            quantization_ratio = calculate_quantization_ratio(x.reshape(-1))
            logger.record(k=k, f=f, x=x, fx=fx, Rx=Rx, Fx=fx + lambda_ * Rx, M=M, residue=residue, quantization_ratio=quantization_ratio, quantized_fx=quantized_fx)

        # gradient descent step
        # eta = 0.2 * eta0 / np.log(k + 1)
        eta = eta0 / np.power(k, 0.1)
        gamma_inv += eta
        y -= eta * gx
        # proximal step
        x = R.prox_mapping(y, lambda_ * gamma_inv)
        # diagonalB = np.ones_like(x)  # Assuming args.diagonalB is not provided
        # x = R.scaled_prox_mapping(10 * L0 * diagonalB, y, gx, lambda_ * gamma_inv)
        fx = f.func(x)

    # maximum number of iterations reached
    print('gamma: ', gamma_inv)
    if logger.status != 'O' and logger.status != 'L':
        logger.status = 'M'

    logger.update_solution(x, M, lambda_, k)
    return logger
