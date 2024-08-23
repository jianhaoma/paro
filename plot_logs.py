import numpy as np
import os
from algo_options import parse_args
import matplotlib.pyplot as plt

def plot_logs(x_gen, Fx_min, lambda0, print_eps, full_path, *loggers):
    args = parse_args()
    # establish new folders according the dimension n
    n = x_gen.shape[0]
    N = len(loggers)
    print('N: ', N)

    line_specs = ['b--', 'k-', 'r-.', 'g-', 'm-', 'c-', 'y-']
    marker_specs = ['bx', 'r+', 'g*', 'mv', 'cs', 'c^', 'y.']

    # plot the solution vector
    plt.figure()
    plt.plot(x_gen, 'ko', label='TRUE') # ground truth
    for i in range(N):
        logger = loggers[i]
        plt.plot(logger.x, marker_specs[i], label=logger.algo_name)
    plt.xlabel('coordinates i')
    plt.ylabel('values x(i)')
    for i in range(args.k):
        plt.plot(i * np.ones(len(logger.x)), 'k')
        plt.plot(-i * np.ones(len(logger.x)), 'k')
    plt.legend(loc='best')
    file_path = os.path.join(full_path, "solution.pdf")
    plt.savefig(file_path)

    # plot F(x)-F_min versus t
    plt.figure()
    for i in range(N):
        logger = loggers[i]
        plt.plot(logger.idx_all, np.array(logger.Fx_all) - Fx_min, line_specs[i], linewidth=2, label=logger.algo_name)
    plt.xlabel('$t$')
    plt.ylabel('$F_{\\lambda}(x_t)-F_{\\lambda}^*$')
    plt.yscale('log')
    plt.legend(loc='best')
    file_path = os.path.join(full_path, "obj_k.pdf")
    plt.savefig(file_path)

    # plot fx versus t
    plt.figure()
    for i in range(N):
        logger = loggers[i]
        plt.plot(logger.idx_all, logger.fx_all, line_specs[i], linewidth=2, label=logger.algo_name)
    plt.xlabel('$t$')
    plt.ylabel('$f(x_{t})$')
    plt.yscale('log')
    plt.legend()
    file_path = os.path.join(full_path, "f_k.pdf")
    plt.savefig(file_path)

    # plot quantized_fx versus t
    plt.figure()
    for i in range(N):
        logger = loggers[i]
        plt.plot(logger.idx_all, logger.quantized_fx_all, line_specs[i], linewidth=2, label=logger.algo_name)
    plt.xlabel('$t$')
    plt.ylabel('$f(Q(x_{t}))$')
    plt.yscale('log')
    plt.legend()
    file_path = os.path.join(full_path, "quantized_f_k.pdf")
    plt.savefig(file_path)

    # plot quantized_fx versus t
    plt.figure()
    for i in range(N):
        logger = loggers[i]
        plt.plot(logger.idx_all, logger.quantization_ratio_all, line_specs[i], linewidth=2, label=logger.algo_name)
    plt.xlabel('$t$')
    plt.ylabel('quantization ratio')
    plt.yscale('log')
    plt.legend()
    file_path = os.path.join(full_path, "quantization_ratio.pdf")
    plt.savefig(file_path)

    