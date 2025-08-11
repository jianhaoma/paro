import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimization for PAR-regularized problems"
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument(
        "--recording",
        action="store_true",
        default=True,
        help="Enable solution history recording",
    )
    parser.add_argument("--loss", type=str, default="l2", help="Loss function")
    parser.add_argument("--regularizer", type=str,
                        default="convex", help="Regularizer")
    parser.add_argument(
        "--line-search",
        action="store_true",
        help="Enable line search in proximal gradient and ADMM",
    )
    parser.add_argument("--no-line-search",
                        action="store_false", dest="line_search")
    parser.set_defaults(line_search=True)
    parser.add_argument(
        "--stochastic", action="store_true", help="Use stochastic gradient"
    )
    parser.add_argument("--no-stochastic",
                        action="store_false", dest="stochastic")
    parser.set_defaults(stochastic=False)
    parser.add_argument(
        "--stopbyKKT",
        action="store_true",
        default=False,
        help="Use KKT condition for stopping criterion",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-5, help="Tolerance for stopping criterion"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=5000,
        help="Maximum number of iterations per stage",
    )
    parser.add_argument(
        "--gamma_inc",
        type=float,
        default=2.0,
        help="Multiplicative constant to increase Lipschitz constant",
    )
    parser.add_argument(
        "--gamma_dec",
        type=float,
        default=2.0,
        help="Dividing constant to decrease Lipschitz constant",
    )
    parser.add_argument(
        "--Lipsc_min",
        type=float,
        default=0.0,
        help="Minimum local Lipschitz constant for line search",
    )
    parser.add_argument(
        "--Lipsc_max",
        type=float,
        default=1e8,
        help="Maximum local Lipschitz constant for line search",
    )

    # generate the arguments
    args = parser.parse_args()

    # assert the arguments
    if args.tolerance <= 0:
        raise ValueError("Tolerance should be a small positive real number")

    if args.max_iters <= 0:
        raise ValueError("Maximum iterations should be a positive integer")

    if args.gamma_inc <= 1.0:
        raise ValueError("Gamma increase should be larger than 1.0")

    if args.gamma_dec <= 1.0:
        raise ValueError("Gamma decrease should be larger than 1.0")

    if args.Lipsc_min < 0.0:
        raise ValueError("Minimum Lipschitz constant should be nonnegative")

    if args.Lipsc_max <= 0.0:
        raise ValueError("Maximum Lipschitz constant should be positive")

    return args
