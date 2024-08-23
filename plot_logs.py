import numpy as np
import os
import matplotlib.pyplot as plt

def plot_logs(x_gen, Fx_min, lambda0, print_eps, full_path, *loggers):
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
    for i in range(5):
        plt.plot(i * np.ones(len(logger.x)), 'k')
        plt.plot(-i * np.ones(len(logger.x)), 'k')
    plt.legend(loc='best')
    file_path = os.path.join(full_path, "solution.pdf")
    plt.savefig(file_path)

    # plot F(x)-F_min versus k
    plt.figure()
    for i in range(N):
        logger = loggers[i]
        plt.plot(logger.idx_all, np.array(logger.Fx_all) - Fx_min, line_specs[i], linewidth=2, label=logger.algo_name)
    plt.xlabel('k')
    plt.ylabel('Fx-Fmin')
    plt.yscale('log')
    plt.legend(loc='best')
    file_path = os.path.join(full_path, "obj_k.pdf")
    plt.savefig(file_path)

    plt.figure()
    for i in range(N):
        logger = loggers[i]
        plt.plot(logger.idx_all, logger.fx_all, linewidth=2, label=logger.algo_name)
        # plt.semilogy(logger.idx_all, logger.quantized_fx_all, linewidth=2)
    plt.xlabel('k')
    plt.ylabel('fx')
    plt.yscale('log')
    plt.legend()
    file_path = os.path.join(full_path, "f_k.pdf")
    plt.savefig(file_path)

    # # plot F(x)-F_min versus mvCounts
    # plt.figure()
    # for i in range(N):
    #     logger = loggers[i]
    #     plt.semilogy(logger.nAx_all, np.array(logger.Fx_all) - Fx_min, line_specs[i], linewidth=2)
    # plt.xlabel('nmv')
    # plt.ylabel('F(x)-Fmin')
    # plt.legend(['PG', 'PGH'])
    # plt.savefig('figure/pgh_obj_mvc.pdf')

    # # plot NNZs versus k
    # plt.figure()
    # for i in range(N):
    #     logger = loggers[i]
    #     plt.plot(logger.idx_all, logger.NNZ_all, line_specs[i], linewidth=2)
    # plt.xlabel('k')
    # plt.ylabel('NNZs')
    # plt.legend(['PG', 'PGH'])
    # plt.savefig('figure/pgh_nnz_k.pdf')

    # # plot optimality residue versus k
    # plt.figure()
    # for i in range(N):
    #     logger = loggers[i]
    #     plt.semilogy(logger.idx_all, logger.res_all, line_specs[i], linewidth=2)
    # plt.xlabel('k')
    # plt.ylabel('residue')
    # plt.legend(['PG', 'PGH'])
    # plt.savefig('figure/pgh_res_k.pdf')

    # # plot Lipschitz constant versus k
    # plt.figure()
    # for i in range(N):
    #     logger = loggers[i]
    #     plt.plot(logger.idx_all, logger.Lf_all, line_specs[i], linewidth=2)
    # plt.xlabel('k')
    # plt.ylabel('Lipschitz')
    # plt.legend(['PG', 'PGH'])
    # plt.savefig('figure/pgh_Lip_k.pdf')

    # # plot recovery error
    # plt.figure()
    # for i in range(N):
    #     logger = loggers[i]
    #     plt.semilogy(logger.idx_all, logger.rcv_all, line_specs[i], linewidth=2)
    # plt.xlabel('k')
    # plt.ylabel('rcv')
    # plt.legend(['PG', 'PGH'])
    # plt.savefig('figure/pgh_rcv_k.pdf')

    # # plot number of iterations for homotopy methods
    # itr_specs = ['bo', 'ko-', 'r*', 'gs-', 'ms', 'c^', 'y.']
    # plt.figure()
    # for i in range(N):
    #     logger = loggers[i]
    #     if logger.algo_name == 'PGH':
    #         plt.semilogx(lambda0 / np.array(logger.lambdas), logger.n_iters, itr_specs[i],
    #                      linewidth=1.5, markersize=10)
    # plt.xlabel('lambda0/lambda')
    # plt.ylabel('niters')
    # plt.legend(['PG', 'PGH'])
    # plt.savefig('figure/pgh_itr_lambda.pdf')
