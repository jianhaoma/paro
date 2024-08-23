import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Proximal Gradient Method')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--recording', action='store_true', default=True,
                        help='Enable solution history recording')
    parser.add_argument('--loss', type=str, default='l2', help='Loss function')
    parser.add_argument('--regularizer', type=str, default='convex', help='Regularizer')
    parser.add_argument("--line-search", action="store_true", help="Enable line search in proximal gradient method")
    parser.add_argument("--no-line-search", action="store_false", dest="line_search")
    parser.set_defaults(line_search=True)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic gradient")
    parser.add_argument("--no-stochastic", action="store_false", dest="stochastic")
    parser.set_defaults(stochastic=False)
    parser.add_argument('--stopbyKKT', action='store_true', default=False,
                        help='Use KKT condition for stopping criterion')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                        help='Tolerance for stopping criterion')
    parser.add_argument('--max_iters', type=int, default=5000,
                        help='Maximum number of iterations per stage')
    parser.add_argument('--gamma_inc', type=float, default=2.0,
                        help='Multiplicative constant to increase Lipschitz constant')
    parser.add_argument('--gamma_dec', type=float, default=2.0,
                        help='Dividing constant to decrease Lipschitz constant')
    parser.add_argument('--Lipsc_min', type=float, default=0.0,
                        help='Minimum local Lipschitz constant for line search')
    parser.add_argument('--Lipsc_max', type=float, default=1e8,
                        help='Maximum local Lipschitz constant for line search')
    parser.add_argument('--delta', type=float, default=0.2,
                        help='Relative precision in homotopy method')
    parser.add_argument('--eta', type=float, default=0.7,
                        help='Multiplicative constant to decrease regularization parameter')
    parser.add_argument('--cgm_ratio', type=float,
                        default=0.1, help='CGM ratio')
    parser.add_argument('--kappa_lbd', type=float,
                        default=2.0, help='Kappa lambda')
    parser.add_argument('--k', type=int, default=2, help='quantization range')
    parser.add_argument('--mucvx_inc', type=float, default=1.0,
                        help='Convexity parameter increase factor')
    parser.add_argument('--mucvx_dec', type=float, default=10.0,
                        help='Convexity parameter decrease factor')

    # generate the arguments
    args = parser.parse_args()

    # assert the arguments
    if args.tolerance <= 0:
        raise ValueError('Tolerance should be a small positive real number')

    if args.max_iters <= 0:
        raise ValueError('Maximum iterations should be a positive integer')

    if args.gamma_inc <= 1.0:
        raise ValueError('Gamma increase should be larger than 1.0')

    if args.gamma_dec <= 1.0:
        raise ValueError('Gamma decrease should be larger than 1.0')

    if args.Lipsc_min < 0.0:
        raise ValueError('Minimum Lipschitz constant should be nonnegative')

    if args.Lipsc_max <= 0.0:
        raise ValueError('Maximum Lipschitz constant should be positive')

    if args.delta >= 1 or args.delta <= 0:
        raise ValueError('Delta should be in the interval (0, 1)')

    if args.eta >= 1 or args.eta <= 0:
        raise ValueError('Eta should be in the interval (0, 1)')

    if args.cgm_ratio <= 0.0 or args.cgm_ratio >= 1.0:
        raise ValueError('CGM ratio should be in the interval (0, 1)')

    if args.kappa_lbd <= 1.0:
        raise ValueError('Kappa lambda should be larger than 1')

    if args.mucvx_inc < 1.0:
        raise ValueError(
            'Convexity parameter increase factor should be not less than 1')

    if args.mucvx_dec <= 1.0:
        raise ValueError(
            'Convexity parameter decrease factor should be larger than 1')

    return args
