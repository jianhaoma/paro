import numpy as np
from algo_options import parse_args
from solution_logger import SolutionLogger


def soft_ste(f, R, lambda_, x0, L0, tol):
    args = parse_args()

    # create logger to store solution history
    logger = SolutionLogger('SOFT STE', x0, L0)

    # initialization
    x = x0
    L = L0
    M = L0
    eta0 = 4 / L0
    eta_sum = 0

    # for optimality test in the first iteration, use fake gradient mapping
    gMx = np.ones_like(x) * 1e10

    fx = f.func(x)
    for k in range(1, args.max_iters + 1):
        # evaluate function and gradient at x
        gx = f.grad_x()

        # compute optimality residue using KKT condition or CGM norm
        if args.stopbyKKT:
            residue = R.optimality_residue(x, gx, lambda_)
        else:
            residue = np.linalg.norm(gMx, np.inf)

        # logger keeps record of solution path
        if args.recording:
            Rx = R.func(x)
            logger.record(k=k, f=f, x=x, fx=fx, Rx=Rx, Fx=fx + lambda_ * Rx, M=M, residue=residue)

        # check for stopping criterion
        if residue <= tol:
            logger.status = 'O'
            break

        # proximal gradient step
        eta = eta0 / k
        eta_sum += eta
        diagonalB = np.ones_like(x)  # Assuming args.diagonalB is not provided
        print(eta_sum * lambda_)
        x1 = R.scaled_prox_mapping(min(eta_sum, 1 / lambda_) * diagonalB, x, gx, lambda_)
        fx1 = f.func(x1)

        # update solution
        x = x1
        fx = fx1

    # maximum number of iterations reached
    if logger.status != 'O' and logger.status != 'L':
        logger.status = 'M'

    # record the latest solution vector and number of iterations
    logger.update_solution(x, M, lambda_, k)

    return logger
