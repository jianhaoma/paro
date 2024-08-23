import numpy as np
import math
from regularizer import Regularizer


class CONVEX_PAR(Regularizer):
    def __init__(self, a, b, q):
        self.mu = 0  # strong convexity parameter
        self.a = a # slopes of the pieces
        self.b = b # intercepts of the pieces
        self.q = q # breakpoints

    def convexity_parameter(self):
        return self.mu

    @staticmethod
    def psi_1d(x, a, b, q):
        if np.abs(x) <= q[-1]:
            return np.max(a * (np.abs(x) - q) + b)
            # remark: we can use a binary search to find the index of the breakpoint, but it is unnecessary if the number of the breakpoints is limited
        else:
            return 1e10

    def func(self, x):
        vec_psi_1d = np.vectorize(CONVEX_PAR.psi_1d, excluded=['a', 'b', 'q'])
        return np.sum(vec_psi_1d(x=x, a=self.a, b=self.b, q=self.q))

    @staticmethod
    def prox_psi_1d(z, lam, a, b, q):
        m = len(a)  # Number of pieces in the piecewise function
        a = np.append(a, 0)
        for k in range(m - 1):
            if np.abs(z) <= lam * a[k] + q[k] and np.abs(z) >= lam * a[k - 1] + q[k]:
                return np.sign(z) * q[k]
            elif np.abs(z) >= lam * a[k] + q[k] and np.abs(z) <= lam * a[k] + q[k + 1]:
                return z - np.sign(z) * lam * a[k]
        return np.sign(z) * q[-1]

    def prox_mapping(self, z, lam):
        """Return argmin_x { (1/2)||x-z||_2^2 + lambda*R(x) }"""
        vec_prox_psi_1d = np.vectorize(CONVEX_PAR.prox_psi_1d, excluded=['a', 'b', 'q'])
        return vec_prox_psi_1d(x=x, lam=lam, a=self.a, b=self.b, q=self.q)

    def scaled_prox_mapping(self, b, z, g, lam):
        """Return argmin_x {sum_i [(1/2)bi*(xi-zi)^2 + gi*xi] + lambda*R(x)}"""
        vec_prox_psi_1d = np.vectorize(CONVEX_PAR.prox_psi_1d, excluded=['a', 'b', 'q'])
        return vec_prox_psi_1d(z=z - g / b, lam=lam / b, a=self.a, b=self.b, q=self.q)

    def optimality_residue(self, x, f_grad, lam):
        pass


'''TODO: implement the quasiconvex PAR'''
class QUASICONVEX_PAR(Regularizer):
    def __init__(self):
        self.mu = 0  # strong convexity parameter

    def convexity_parameter(self):
        return self.mu

    @staticmethod
    def psi_1d(x):
        '''TBD'''
        return 1.0

    def func(self, x):
        x = x.flatten()
        return sum([QUASICONVEX_PAR.psi_1d(x[i]) for i in range(len(x))])

    @staticmethod
    def prox_psi_1d(z, lam):
        if z < 0:
            return -QUASICONVEX_PAR.prox_psi_1d(-z, lam)
        z_floor = np.floor(z)
        diff = z - z_floor
        if lam <= 1:
            if diff < lam:
                return z_floor
            elif diff < 0.5 * (1 + lam):
                return z_floor + diff - lam
            else:
                return z_floor + diff
        else:
            return np.floor(z - 0.5 * (lam - 1))

    def prox_mapping(self, z, lam):
        """Return argmin_x { (1/2)||x-z||_2^2 + lambda*R(x) }"""
        return np.array([QUASICONVEX_PAR.prox_psi_1d(z[i], lam) for i in range(len(z))]).reshape(z.shape)

    def scaled_prox_mapping(self, b, z, g, lam):
        """Return argmin_x {sum_i [(1/2)bi*(xi-zi)^2 + gi*xi] + lambda*R(x)}"""
        m = len(z)
        return np.array([QUASICONVEX_PAR.prox_psi_1d(z[i] - g[i] / b[i], lam / b[i]) for i in range(m)]).reshape(z.shape)

    def optimality_residue(self, x, f_grad, lam):
        pass


class NONCONVEX_PAR(Regularizer):
    def __init__(self):
        self.mu = 0  # strong convexity parameter

    def convexity_parameter(self):
        return self.mu

    @staticmethod
    def psi_1d(x):
        n = math.floor(x)
        # return the value according to whether n is even or odd
        return x - n if n % 2 == 0 else n + 1 - x

    def func(self, x):
        return 0.5 * np.sum(np.vectorize(NONCONVEX_PAR.psi_1d)(2 * x))

    @staticmethod
    def prox_psi_1d(z, lam):
        if z < 0:
            return -NONCONVEX_PAR.prox_psi_1d(-z, lam)
        z_floor = np.floor(z)
        return max(z_floor, z - lam) if z - z_floor < 0.5 else min(z_floor + 1, z + lam)

    def prox_mapping(self, z, lam):
        """Return argmin_x { (1/2)||x-z||_2^2 + lambda*R(x) }"""
        return np.vectorize(NONCONVEX_PAR.prox_psi_1d)(z, lam)

    def scaled_prox_mapping(self, b, z, g, lam):
        """Return argmin_x {sum_i [(1/2)bi*(xi-zi)^2 + gi*xi] + lambda*R(x)}"""
        return np.vectorize(NONCONVEX_PAR.prox_psi_1d)(z - g / b, lam / b)

    def optimality_residue(self, x, f_grad, lam):
        pass
