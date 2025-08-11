import numpy as np
from algo_options import parse_args
from solution_logger import SolutionLogger


def ADMM(f, R, lambda_, x0, L0, tol):
    args = parse_args()
    rho = 1
    rho_min = 1e-6
    rho_max = 1e6
    mu = 10
    tau_incr = 2
    tau_decr = 2
    abs_tol = 1e-7
    rel_tol = 1e-5

    # create logger to store solution history
    logger = SolutionLogger("ADMM", x0, L0)

    # initialization
    d, _ = x0.shape
    x = np.zeros((d, 1))
    z = np.zeros((d, 1))
    u = np.zeros((d, 1))

    fx = f.func(x)
    for k in range(1, args.max_iters + 1):
        # compute next iterates
        x1 = f.global_min_x(z - u, rho)
        z1 = R.prox_mapping(x1 + u, lambda_ / rho)
        u1 = u + x1 - z1
        fx1 = f.func(x1)

        # compute optimality residue
        residue = np.linalg.norm(x1 - z1)
        residue_s = rho * np.linalg.norm(z1 - z)

        # update solution
        x = x1
        z = z1
        u = u1
        fx = fx1

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
                M=L0,
                residue=residue,
                quantized_fx=quantized_fx,
                quantization_rate=quantization_rate
            )

        # Tolerances
        eps_pri = np.sqrt(d) * abs_tol + rel_tol * max(
            np.linalg.norm(x), np.linalg.norm(z)
        )
        eps_dual = np.sqrt(d) * abs_tol + rel_tol * np.linalg.norm(rho * u)

        if args.line_search:
            if residue > mu * residue_s:
                rho_new = min(rho * tau_incr, rho_max)
            elif residue_s > mu * residue:
                rho_new = max(rho / tau_decr, rho_min)
            else:
                rho_new = rho

            # Scale dual variable if rho has changed
            if rho_new != rho:
                u *= rho / rho_new
                rho = rho_new

        # check for stopping criterion
        if residue <= eps_pri and residue_s <= eps_dual:
            logger.status = "O"
            break

    # maximum number of iterations reached
    if logger.status != "O" and logger.status != "L":
        logger.status = "M"

    # record the latest solution vector and number of iterations
    logger.update_solution(x, L0, lambda_, k)

    return logger


