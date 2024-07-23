import numpy as np
import casadi as ca

class GPModel:
    '''
    GP Model. Assumes Paramters are already known\\
    :param kernel: callable: Kernel function that takes as input 2 arrays, of proper dimension
    and return the corresponding kernel matrix entry 
    :param noise_var: float: Noise variance used in the predictive distribution computations
    :param input_dim: int: Dimension of the input data (has to match with the dimension of the 
    input to the kernel function)
    :param: output_dim: Dimension of the output prediction
    :param: window_size: Dimension of the window of active window used to make predictions
    '''
    def __init__(
            self, 
            kernel: callable, 
            noise_var: float,
            input_dim: int, 
            output_dim: int,
            window_size: int
        ):

        self.__kernel = kernel
        self.__noise_var = noise_var
        self.__input_dim = input_dim
        self.__output_dim = output_dim
        self.__window_size = window_size
        self.__inputs = []
        self.__labels = []
        self.__size = 0
        self.__K = np.zeros((window_size,window_size))
        self.__Kxx = ca.SX.sym('Kxx',window_size)
        self.__new_entry = np.zeros(window_size)

    def update(self, input, label):
        '''
        Update the model by adding a new point for the prediction. In case the number of points is
        greate then the window size, the least recent point id discarded
        '''
        if self.__size >= self.__window_size:
            # Compute the updated Kernel matrix
            for k in range(self.__size):
                K_xx = self.__kernel(input,self.__inputs[k])
                self.__new_entry[k] = K_xx
            k = self.__kernel(input,input)
            self.__K = np.block([
                [self.__K, self.__new_entry.reshape((50,1))],
                [self.__new_entry.reshape((1,50)), k]
            ])
            self.__K = self.__K[1:,1:]
            # Update Inputs and Labels
            self.__inputs.append(input.copy())
            self.__inputs.pop(0)
            self.__labels.append(label.copy())
            self.__labels.pop(0)
        else:
            # Compute the updated Kernel matrix
            for k in range(self.__size):
                K_xx = self.__kernel(input,self.__inputs[k])
                self.__K[self.__size,k] = K_xx
                self.__K[k,self.__size] = K_xx
            self.__K[self.__size,self.__size] = self.__kernel(input,input)
            # Update Inputs and Labels
            self.__inputs.append(input.copy())
            self.__labels.append(label.copy())
            # Update the size
            self.__size += 1

    def __call__(self, x):
        labels = np.array(self.__labels)
        for k in range(self.__window_size):
            self.__Kxx[k] = self.__kernel(x,self.__inputs[k])
        K_inv = np.linalg.inv(self.__K+self.__noise_var*np.eye(self.__window_size))
        # Compute Predictive Mean
        mean = self.__Kxx.T@K_inv@labels
        # Compute Predictive Variance
        covar = self.__kernel(x,x)-self.__Kxx.T@K_inv@self.__Kxx

        return mean, covar
    
    def __len__(self):
        return self.__size