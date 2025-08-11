from loss import LossFunction
import numpy as np


class SquaredLoss(LossFunction):
    def __init__(self, A, b):
        self.A = A  # design matrix of size n x p
        self.b = b  # response vector of size n x 1
        self.r = np.zeros_like(b)  # residual vector
        self.mvCount = 0  # number of matrix vector multiplications
        self.n = A.shape[0]  # number of samples
        self.p = A.shape[1]  # number of features

    def func(self, x):
        '''calculate the function value at x $(1/2n)*\|Ax-b\|^2$'''
        self.r = self.A @ x - self.b
        fx = (0.5 / self.n) * np.linalg.norm(self.r) ** 2
        self.mvCount += 1
        return fx

    def grad_x(self):
        '''calculate the gradient at the last x that func was called $(1/n)A^T(Ax-b)$'''
        gx = self.A.T @ self.r / self.n
        self.mvCount += 1
        return gx

    def grad(self, x):
        '''calculate the gradient at x $(1/n)A^T(Ax-b)$'''
        self.r = self.A @ x - self.b
        gx = self.A.T @ self.r / self.n
        self.mvCount += 2
        return gx

    def stochastic_grad(self, x):
        '''calculate a stochastic subgradient at x $(a_i.T @ x - b_i) * a_i$'''
        index = np.random.choice(self.n)
        a_i = self.A[index, :].reshape(-1, 1)
        self.r = a_i.T @ x - self.b[index]
        gx = self.r * a_i
        return gx

    def func_grad(self, x):
        '''calculate both function value and gradient at x'''
        self.r = self.A @ x - self.b
        fx = 0.5 * np.linalg.norm(self.r) ** 2 / self.n
        gx = self.A.T @ self.r / self.n
        self.mvCount += 2
        return fx, gx

    def global_min_x(self, x, rho):
        '''Return global minimum point of the loss function $(A^TA/n + rho*I)^{-1}(A^Tb/n + rho*x)$'''
        L = np.linalg.cholesky(self.A.T @ self.A /
                               self.n + rho * np.eye(self.p))
        U = L.T
        q = self.A.T @ self.b / self.n + rho * x
        y = np.linalg.solve(L, q)
        return np.linalg.solve(U, y)

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
