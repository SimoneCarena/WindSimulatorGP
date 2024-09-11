import numpy as np

class _RBFKernelBase:
    def __init__(self, lengthscale, outputscale):
        self.__lengthscale = lengthscale
        self.__outputscale = outputscale

    def __call__(self, x1, x2):
        return self.__outputscale*np.exp(
            -((x1-x2).T@(x1-x2))/(2*self.__lengthscale**2)
        )

class _RBFKernelDerivative:
    def __init__(self, lengthscale, outputscale):
        self.__lengthscale = lengthscale
        self.__outputscale = outputscale
    
    def __call__(self, x1, x2):
        return -self.__outputscale*np.exp(
            -((x1-x2).T@(x1-x2))/(2*self.__lengthscale**2)
        )*(x1-x2)/self.__lengthscale**2

class RBFKernel:
    def __init__(self, lengthscale, outputscale):
        self.__lengthscale = lengthscale
        self.__outputscale = outputscale

    def get_kernel(self):
        return _RBFKernelBase(self.__lengthscale,self.__outputscale)
    
    def get_kernel_derivative(self):
        return _RBFKernelDerivative(self.__lengthscale,self.__outputscale)