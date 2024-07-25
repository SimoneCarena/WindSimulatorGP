import numpy as np

class RBFKernel:
    def __init__(self, lengthscale, outputscale):
        self.__lengthscale = lengthscale
        self.__outputscale = outputscale

    def __call__(self, x1, x2):
        return self.__outputscale*np.exp(
            -((x1-x2).T@(x1-x2))/(2*self.__lengthscale**2)
        )