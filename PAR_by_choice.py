import numpy as np
from algo_options import parse_args
from squared_loss import SquaredLoss
from logistic_loss import LogisticLoss
from prox_grad import prox_grad
from homotopy import homotopy
from parq import parq
from proxquant import proxquant
from ste import ste
from soft_ste import soft_ste
from new_ste import new_ste


def PAR_by_choice(A, b, f, R, lam, algo):
    args = parse_args()
    _, d = A.shape

    # initialization with zeros (it can be relaced with other initializations)
    x0 = np.zeros((d, 1))
    try:
        L0 = max(f.diag_Hessian())
    except:
        L0 = 1
    lam0 = f.max_l1_regularization()

    if algo == 'PG':
        logger = prox_grad(f, R, lam, x0, L0, args.tolerance)
    elif algo == 'PGH':
        logger = homotopy(prox_grad, f, R, lam0, lam, x0, L0)
    elif algo == 'PARQ':
        logger = parq(f, R, lam, x0, L0, args.tolerance)
    elif algo == 'STE':
        logger = ste(f, R, lam, x0, L0, args.tolerance)
    elif algo == 'new STE':
        logger = new_ste(f, R, lam, x0, L0, args.tolerance)
    elif algo == 'quasiconvex PARQ':
        logger = parq(f, R, lam, x0, L0, args.tolerance)
        logger.algo_name = 'quasiconvex PARQ'
    elif algo == 'nonconvex PARQ':
        logger = parq(f, R, lam, x0, L0, args.tolerance)
        logger.algo_name = 'nonconvex PARQ'
    elif algo == 'PROXQUANT':
        logger = proxquant(f, R, lam, x0, L0, args.tolerance)
    else:
        raise ValueError(f"Algorithm {algo} is not implemented.")

    return logger
