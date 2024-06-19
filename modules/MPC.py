import numpy as np
import torch

from scipy.optimize import minimize

class MPC:
    def __init__(self, A, B, Q, R, T, predictor, u_min = None, u_max = None):
        # Discretized system matrices
        self.__A = A
        self.__B = B
        # Cost matrices
        self.__Q = Q
        self.__R = R
        # Horizon
        self.__T = T
        # Actuators constraints
        if u_min is not None:
            self.__u_min = u_min
        else:
            self.__u_min = np.array([-np.inf]*len(A))
        if u_max is not None:
            self.__u_max = u_max
        else:
            self.__u_max = np.array([np.inf]*len(A))
        # GP Predictor
        self.__predictor = predictor

    def __cost_function(self, u, x0, x_ref):
        x = x0
        cost = 0.0
        for t in range(self.__T):
            x = self.__A @ x + self.__B @ u[t] + self.__predictor(torch.from_numpy(x)).mean.numpy()
            cost += (x - x_ref[t]).T @ self.Q @ (x - x_ref[t]) + u[t].T @ self.R @ u[t]
        return cost

    def __call__(self, x0, x_ref):
        # Complete the sequence in case the trajectory is completed
        if len(x_ref)<self.__T:
            l = len(x_ref)
            for _ in range(self.__T-l):
                x_ref.append(x_ref[-1])
        
        u0 = np.zeros((self.__T, self.__B.shape[1]))  # Initial guess for the inputs
        bounds = [(self.__u_min, self.__u_max) for _ in range(self.__T)]
        result = minimize(self.__cost_function, u0, args=(x0,x_ref,), bounds=bounds)
        if result.success:
            return result.x[0]
        else:
            raise ValueError("MPC optimization failed")

