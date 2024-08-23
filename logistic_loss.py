from loss import LossFunction
import numpy as np


def sigmoid(z):
    return 1. / (1 + np.exp(-z))


class LogisticLoss(LossFunction):
    def __init__(self, A, b):
        self.A = A # design matrix
        self.b = b # response vector
        self.r = np.zeros_like(b) # residual vector
        self.mvCount = 0 # number of matrix-vector multiplications
        self.n = A.shape[0] # number of samples

    def func(self, x):
        '''
        calculate the function value at x
        (1/n) * sum_i log(1 + exp(-b_i * (A_i * x)))
        '''
        h = sigmoid(self.b * self.A @ x)
        self.r = h * self.b - self.b
        fx = -np.mean(np.log(h))
        self.mvCount += 1
        return fx

    def grad_x(self):
        '''
        calculate the gradient at the last x that func was called
        (1/n) * sum_i b_i * A_i * (h_i - 1)
        '''
        gx = self.A.T @ self.r / self.n
        self.mvCount += 1
        return gx

    def grad(self, x):
        '''calculate the gradient at x'''
        h = sigmoid(self.b * self.A @ x)
        r = h * self.b - self.b
        gx = self.A.T @ r / self.n
        self.mvCount += 2
        return gx

    def stochastic_grad(self, x):
        '''calculate a stochastic subgradient at x'''
        # pick a random sample
        index = np.random.choice(self.A.shape[0])
        # calculate the gradient at the sample
        a_i = self.A[index, :].reshape(-1, 1)
        r_i = sigmoid(self.b[index] * a_i.T @ x) * self.b[index] - self.b[index]
        gx = r_i * a_i
        self.mvCount += 2
        return gx

    def func_grad(self, x):
        '''calculate both function value and gradient at x'''
        h = sigmoid(self.b * self.A @ x)
        self.r = h * self.b - self.b
        fx = -np.mean(np.log(h))
        gx = self.A.T @ self.r / self.n
        self.mvCount += 2
        return fx, gx

    def total_mvCount(self):
        return self.mvCount

    def reset_mvCount(self):
        self.mvCount = 0

    def max_l1_regularization(self):
        '''Return max regularization weight for l1'''
        l1max = np.max(np.abs(self.A.T @ self.b)) / self.n
        return l1max

    def diag_Hessian(self):
        '''
        Return diagonal of Hessian matrix (for quasi-Newton direction)
        Currently undefined for class objects other than numerical matrix
        '''
        d = np.sum(self.A * self.A, axis=0) / self.n
        return d

    def initial_Lipschitz(self):
        '''Return initial Lipschitz constant'''
        m, n = self.A.shape
        unit1 = np.zeros(n)
        unit1[0] = 1
        L0 = np.linalg.norm(self.A @ unit1) ** 2 / m
        return L0
