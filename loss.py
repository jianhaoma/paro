from abc import ABC, abstractmethod


class LossFunction(ABC):
    @abstractmethod
    def func(self, x):
        '''Return function value f(x)'''
        pass

    @abstractmethod
    def grad_x(self):
        '''Return gradient at the last x that func was called'''
        pass

    @abstractmethod
    def grad(self, x):
        '''Return gradient at x'''
        pass

    def stochastic_grad(self, x):
        '''Return stochastic gradient at x'''
        pass

    @abstractmethod
    def func_grad(self, x):
        '''Return both function value and gradient at x'''
        pass

    @abstractmethod
    def global_min_x(self, x, rho):
        '''Return global minimum point of the loss function'''
        pass
