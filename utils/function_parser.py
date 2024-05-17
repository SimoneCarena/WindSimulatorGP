import numpy as np

from utils.exceptions import InvalidFunctionException

class _DefaultFunction():
    def __init__(self,v0):
        self.__v0 = v0
    def __call__(self,t):
        return self.__v0
    
class _SinFunction:
    def __init__(self,v0,f,phi):
        self.__v0 = v0
        self.__f = f
        self.__phi = phi
    def __call__(self, t):
        return self.__v0*np.sin(2*np.pi*self.__f*t+self.__phi)
    
class _SquareFunction:
    def __init__(self,v0,f):
        self.__v0 = v0
        self.__f = f
    def __call__(self,t):
        return self.__v0*np.sign(np.sin(2*np.pi**self.__f*t))


def parse_generator(generator):
    parameters = generator['parameters']
    if generator['function'] == 'sin':
        v0 = parameters['v0']
        f = parameters['frequency']
        phi = np.deg2rad(parameters['phase'])
        generator_function = _SinFunction(v0,f,phi)
        return generator_function
    elif generator['function'] == 'cos':
        v0 = parameters['v0']
        f = parameters['frequency']
        phi = np.deg2rad(parameters['phase']+90)
        generator_function = _SinFunction(v0,f,phi)
        return generator_function 
    elif generator['function'] == 'constant':
        v0 = parameters['v0']
        generator_function = _DefaultFunction(v0)
        return generator_function
    elif generator['function'] == 'square':
        v0 = parameters['v0']
        f = parameters['frequency']
        generator_function = _SquareFunction(v0,f)
        return generator_function

    else:
        raise InvalidFunctionException(f'Unexpected function "{generator['function']}" while parsing fans\' data')

