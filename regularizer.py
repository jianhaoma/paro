from abc import ABC, abstractmethod


class Regularizer(ABC):
    @abstractmethod
    def func(self, x):
        """Return function value R(x)"""
        pass

    @abstractmethod
    def prox_mapping(self, z, lambda_):
        """Return argmin_x { (1/2)||x-z||_2^2 + lambda*R(x) }"""
        pass

    @abstractmethod
    def scaled_prox_mapping(self, b, z, g, lambda_):
        """Return argmin_x sum_i [(1/2)bi*(xi-zi)^2 + gi*xi] + lambda*R(x)"""
        pass
