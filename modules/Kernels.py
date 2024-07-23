import numpy as np

class RBFKernel:
    def __init__(self, lengthscale):
        self.__lengthscale = lengthscale

    def __call__(self, x1, x2):
        return np.exp(
            ((x1-x2).T@(x1-x2))/(2*self.__lengthscale**2)
        )