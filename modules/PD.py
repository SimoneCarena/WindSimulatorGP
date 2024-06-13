import numpy as np

class PD:
    '''
    A simple PD controller
    '''
    def __init__(self,Kp,Kd):
        self.Kp = Kp.copy()
        self.Kd = Kd.copy()
        self.upper_bound = np.inf
        self.lower_bound = -np.inf
    
    def step(self, ep, ev):
        '''
        Generate control command
        '''
        output = self.Kp@ep + self.Kd@ev

        return np.clip(output,self.lower_bound,self.upper_bound)
