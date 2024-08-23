import numpy as np
from algo_options import parse_args
from solution_logger import SolutionLogger


def prox_grad(f, R, lam, x0, L0, tol):
    args = parse_args()

    # create logger to store solution history
    logger = SolutionLogger('PG', x0, L0)

    # initialization
    x = x0
    L = L0
    M = L0

    # for optimality test in the first iteration, use fake gradient mapping
    gMx = np.ones_like(x) * 1e10

    fx = f.func(x)
    for k in range(1, args.max_iters + 1):
        # calculate the (stochastic) gradient at x
        if args.stochastic:
            gx = f.stochastic_grad(x)
        else:
            gx = f.grad_x()

        # compute optimality residue using the norm of the composite gradient mapping
        residue = np.linalg.norm(gMx, ord=np.inf)

        # logger keeps record of solution path
        if args.recording:
            Rx = R.func(x)
            quantized_x = np.round(x)
            quantized_fx = f.func(quantized_x)
            logger.record(k=k, f=f, x=x, fx=fx, Rx=Rx, Fx=fx + lam*Rx, M=M, residue=residue, quantized_fx=quantized_fx)

        # check for stopping criterion
        if residue <= tol:
            logger.status = 'O'
            break

        # compute next iterate with or without line search
        if args.line_search:
            x1, M, gMx, fx1 = line_search(f, R, lam, x, fx, gx, L, args)

            # reduce Lipschitz constant by fixed factor if possible
            L = max(args.Lipsc_min, M / args.gamma_dec)
        else:
            diagonalB = np.ones_like(x) * 4  # Assuming args.diagonalB is not provided
            x1 = R.scaled_prox_mapping(L0 * diagonalB, x, gx, lam)
            fx1 = f.func(x1)
            dx = x1 - x
            gMx = -L0 * (diagonalB * dx)

        # update solution
        x = x1
        fx = fx1

    # maximum number of iterations reached
    if logger.status != 'O' and logger.status != 'L':
        logger.status = 'M'

    # record the latest solution vector and number of iterations
    logger.update_solution(x, M, lam, k)

    return logger


def line_search(f, R, lam, x, fx, gx, L, args):
    while L <= args.Lipsc_max:
        diagonalB = np.ones_like(x)  # Assuming args.diagonalB is not provided
        x1 = R.scaled_prox_mapping(L * diagonalB, x, gx, lam)
        fx1 = f.func(x1)
        dx = x1 - x
        gMx = -L * (diagonalB * dx)
        if fx1 < fx + np.sum(dx * gx) - 0.5 * np.sum(dx * gMx):
            break
        L = L * args.gamma_inc

    # detect line search failure and throw exception
    if L > args.Lipsc_max:
        raise LineSearchException(f"Lipsc_max = {args.Lipsc_max} reached")

    return x1, L, gMx, fx1
