import numpy as np

class Sphere:
    '''
    Class describing the object moving in the wind field.\\
    The object consist of a sphere of mass m and radius r\\
    The sampling time dt is specified to compute the dynamics of the system
    '''
    def __init__(self, m, r, x0, y0, v0x, v0y, dt):
        self.m = m
        self.r = r
        self.p = np.array([x0,y0],dtype=float)
        self.v = np.array([v0x,v0y],dtype=float)
        self.dt = dt
    
    def move(self,force):
        '''
        Move the object given the force applied to it.\\
        Returns 
        '''
        self.p[0] = self.p[0] + self.dt*self.v[0]
        self.p[1] = self.p[1] + self.dt*self.v[1]
        self.v = self.v + 1/self.m*self.dt*force

        return self.p.copy(), self.v.copy()
