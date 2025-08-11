import numpy as np
from algo_options import parse_args
from solution_logger import SolutionLogger


def acc_prox_grad(f, R, lambda_, x0, L0, tol):
    args = parse_args()

    # create logger to store solution history
    logger = SolutionLogger("acc_PG", x0, L0)

    # initialization
    x = x0
    L = L0
    M = L0

    # for optimality test in the first iteration, use fake gradient mapping
    gMx = np.ones_like(x) * 1e10
    x_prev = x.copy()

    fx = f.func(x)
    for k in range(args.max_iters):
        # calculate the (stochastic) gradient at x
        if args.stochastic:
            gx = f.stochastic_grad(x)
        else:
            gx = f.grad_x()

        # compute optimality residue
        residue = np.linalg.norm(gMx, ord=np.inf)

        # logger keeps record of solution path
        if args.recording:
            Rx = R.func(x)
            quantized_x = R.get_quantized_solution(x)
            quantized_fx = f.func(quantized_x)
            quantization_rate = R.calculate_quantization_rate(x)
            logger.record(
                k=k,
                f=f,
                x=x,
                fx=fx,
                Rx=Rx,
                Fx=fx + lambda_ * Rx,
                M=M,
                residue=residue,
                quantized_fx=quantized_fx,
                quantization_rate=quantization_rate
            )

        # check for stopping criterion, here we use the gradient norm
        if residue <= tol:
            logger.status = "O"
            break

        v = x + (k-2) / (k+1) * (x - x_prev)
        fv = f.func(v)
        gv = f.grad(v)
        # compute next iterate with or without line search
        if args.line_search:
            x1, M, gMx, fx1 = line_search(f, R, lambda_, v, fv, gv, L, args)

            # reduce Lipschitz constant by fixed factor if possible
            L = max(args.Lipsc_min, M / args.gamma_dec)
        else:
            # Assuming args.diagonalB is not provided
            diagonalB = np.ones_like(v)
            x1 = R.scaled_prox_mapping(L * diagonalB, v, gv, lambda_)
            fx1 = f.func(x1)
            dx = x1 - x
            gMx = L * (diagonalB * dx)

        # update solution
        x_prev = x.copy()
        x = x1
        fx = fx1

    # maximum number of iterations reached
    if logger.status != "O" and logger.status != "L":
        logger.status = "M"

    # record the latest solution vector and number of iterations
    logger.update_solution(x, M, lambda_, k)

    return logger


def line_search(f, R, lambda_, x, fx, gx, L, args):
    while L <= args.Lipsc_max:
        diagonalB = np.ones_like(x)  # Assuming args.diagonalB is not provided
        x1 = R.scaled_prox_mapping(L * diagonalB, x, gx, lambda_)
        fx1 = f.func(x1)
        dx = x1 - x
        gMx = L * (diagonalB * dx)
        if fx1 < fx + np.sum(dx * gx) + 0.5 * np.sum(dx * gMx):
            break
        L = L * args.gamma_inc

    # detect line search failure and throw exception
    if L > args.Lipsc_max:
        diagonalB = np.ones_like(x)  # Assuming args.diagonalB is not provided
        x1 = R.scaled_prox_mapping(L * diagonalB, x, gx, lambda_)
        fx1 = f.func(x1)
        dx = x1 - x
        gMx = L * (diagonalB * dx)
        raise ValueError("Line search failed")

    return x1, L, gMx, fx1
