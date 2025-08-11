from loss import LossFunction
import numpy as np


def sigmoid(z):
    return 1. / (1 + np.exp(-z))


class LogisticLoss(LossFunction):
    def __init__(self, A, b):
        self.A = A  # design matrix of size n x p
        self.b = b  # response vector of size n x 1
        self.r = np.zeros_like(b)  # residual vector of size n x 1
        self.mvCount = 0  # number of matrix-vector multiplications
        self.n = A.shape[0]  # number of samples

    def func(self, x):
        '''
        calculate the function value at x $(-1/n) * sum_i (b_i * log(p_i) + (1 - b_i) * log(1 - p_i))$
        '''
        p = sigmoid(self.A @ x)
        self.r = p - self.b
        fx = -np.mean(self.b * np.log(p) + (1 - self.b) * np.log(1 - p))
        self.mvCount += 1
        return fx

    def grad_x(self):
        '''
        calculate the gradient at the last x that func was called
        (1/n) * A^T * (p - b)
        '''
        gx = self.A.T @ self.r / self.n
        self.mvCount += 1
        return gx

    def grad(self, x):
        '''calculate the gradient at x'''
        p = sigmoid(self.A @ x)
        r = p - self.b
        gx = self.A.T @ r / self.n
        self.mvCount += 2
        return gx

    def stochastic_grad(self, x):
        '''calculate a stochastic subgradient at x'''
        # pick a random sample
        index = np.random.choice(self.A.shape[0])
        # calculate the gradient at the sample
        a_i = self.A[index, :].reshape(-1, 1)
        r_i = sigmoid(a_i.T @ x) - self.b[index]
        gx = r_i * a_i
        self.mvCount += 2
        return gx

    def func_grad(self, x):
        '''calculate both function value and gradient at x'''
        p = sigmoid(self.A @ x)
        self.r = p - self.b
        fx = -np.mean(self.b * np.log(p) + (1 - self.b) * np.log(1 - p))
        gx = self.A.T @ self.r / self.n
        self.mvCount += 2
        return fx, gx

    def total_mvCount(self):
        return self.mvCount

    def reset_mvCount(self):
        self.mvCount = 0

    def initial_Lipschitz(self):
        '''Return initial Lipschitz constant'''
        m, n = self.A.shape
        unit1 = np.zeros(n)
        unit1[0] = 1
        L0 = np.linalg.norm(self.A @ unit1) ** 2 / m
        return L0
