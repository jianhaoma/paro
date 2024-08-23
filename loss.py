from abc import ABC, abstractmethod


class LossFunction(ABC):
    @abstractmethod
    def func(self, x):
        '''Return function value f(x)'''
        pass

    def stochastic_grad(self, x):
        '''Return a stochastic subgradient at x'''
        pass

    @abstractmethod
    def grad_x(self):
        '''Return gradient at the last x that func was called'''
        pass

    @abstractmethod
    def grad(self, x):
        '''Return gradient at x'''
        pass

    @abstractmethod
    def func_grad(self, x):
        '''Return both function value and gradient at x'''
        pass
