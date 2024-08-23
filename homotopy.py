import math
from algo_options import parse_args
from solution_logger import SolutionLogger


def homotopy(algo, f, R, lambda0, lambda_tgt, x0, L0):
    args = parse_args()

    # create logger to store solution history
    logger = SolutionLogger('PGH', x0, L0)

    # initialization
    lam = lambda0
    x = x0
    L = L0

    # calculate number of regularization parameters for continuation
    N_stages = math.floor(
        math.log(lambda0 / lambda_tgt) / math.log(1.0 / args.eta))

    for k in range(1, N_stages + 1):
        lam = args.eta * lam
        epsilon = args.delta * lam

        # solving for each intermediate stage
        logger = algo(f, R, lam, x, L, epsilon)

        x = logger.x
        L = logger.M

        if args.recording:
            logger.concatenate(logger)

    # solving the final stage to precision tol
    logger = algo(f, R, lambda_tgt, x, L, args.tolerance)

    if args.recording:
        logger.concatenate(logger)

    logger.assign_name(logger.algo_name + 'H')

    return logger
