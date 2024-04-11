import numpy as np

class PID:
    '''
    A simple PID controller
    '''
    def __init__(self,Kp,Ki,Kd,dt):
        self.Kp = Kp*np.eye(2,2)
        self.Ki = Ki*np.eye(2,2)
        self.Kd = Kd*np.eye(2,2)
        self.dt = dt
        self.error = np.zeros((2,))
        self.integral = np.zeros((2,))
        self.upper_bound = 1
        self.lower_bound = -1
    
    def step(self, error):
        '''
        Generate control command
        '''
        self.integral += self.dt*error
        derivative = (error-self.error)/self.dt
        output = self.Kp@error + self.Ki@self.integral + self.Kd@derivative
        self.error = error.copy()

        return np.clip(output,self.lower_bound,self.upper_bound)
