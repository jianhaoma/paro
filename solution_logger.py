import numpy as np


class SolutionLogger:
    """
    A logger class to record values of all iterates in an iterative algorithm

    Properties:
        status: 'U':Unknown, 'O':Optimal, 'M':MaxIterReached
        algo_name: name of algorithm that generated this logger
        x:      final solution
        M:      local Lipschitz constant from line search at final solution x
        lambdas: vector of regularization parameters along homotopy path
        n_iters: vector of number of iterations for each lambda on homotopy path
        fx_all:  values of f(x) at all iterates
        Rx_all:  values of R(x) at all iterates
        Fx_all:  values of f(x)+lambda*R(x) at all iterates
        Lf_all:  values of local Lipschitz constants at all iterates
        NNZ_all: numbers of non-zeros (||x||_0) at all iterates
        rcv_all: recovery errors if oroginal signal x_gen is provided
        res_all: optimality residues at all iterates
        idx_all: vector of indices for plotting
        nAx_all: numbers of matrix-vector multiplications
    """

    def __init__(self, algo_name, x0, L0, x_gen=None):
        self.verbose = False
        self.x_gen = x_gen if x_gen is not None else []
        self.status = "U"
        self.algo_name = algo_name
        self.x = x0
        self.M = L0
        self.lambdas = []
        self.n_iters = []
        self.fx_all = []
        self.Rx_all = []
        self.Fx_all = []
        self.Lf_all = []
        self.NNZ_all = []
        self.rcv_all = []
        self.res_all = []
        self.idx_all = []
        self.nAx_all = []
        self.quantization_rate_all = []
        self.quantized_fx_all = []

    def assign_name(self, name):
        self.algo_name = name

    def record(
        self, k, f, x, fx, Rx, Fx, M, residue, quantized_fx=0.0, quantization_rate=0.0
    ):
        self.fx_all.append(fx)
        self.quantized_fx_all.append(quantized_fx)
        self.quantization_rate_all.append(quantization_rate)
        self.Rx_all.append(Rx)
        self.Fx_all.append(Fx)
        self.Lf_all.append(M)
        self.NNZ_all.append(np.sum(x != 0))
        self.rcv_all.append(np.linalg.norm(x - self.x_gen))
        self.res_all.append(residue)
        self.idx_all.append(k - 1)
        self.nAx_all.append(f.total_mvCount())

    def update_solution(self, x, M, lambda_, k):
        self.x = x
        self.M = M
        self.lambdas.append(lambda_)
        self.n_iters.append(k)

    def concatenate(self, nextlog):
        self.status = nextlog.status
        self.x = nextlog.x
        self.M = nextlog.M
        self.fx_all.extend(nextlog.fx_all)
        self.quantization_rate_all.extend(nextlog.quantization_rate_all)
        self.Rx_all.extend(nextlog.Rx_all)
        self.Fx_all.extend(nextlog.Fx_all)
        self.Lf_all.extend(nextlog.Lf_all)
        self.NNZ_all.extend(nextlog.NNZ_all)
        self.rcv_all.extend(nextlog.rcv_all)
        self.res_all.extend(nextlog.res_all)
        self.nAx_all.extend(nextlog.nAx_all)
        self.lambdas.extend(nextlog.lambdas)
        self.n_iters.extend(nextlog.n_iters)

        idx_length = len(self.idx_all)
        idx_shift = 0 if idx_length == 0 else self.idx_all[idx_length - 1]
        self.idx_all.extend([idx + idx_shift for idx in nextlog.idx_all])
