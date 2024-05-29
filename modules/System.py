import numpy as np

class System:
    '''
    Class describing the object moving in the wind field.\\
    The object consist of a sphere of mass m and radius r\\
    The sampling time dt is specified to compute the dynamics of the system
    '''
    def __init__(self, m, r, dt):
        self.r = r
        self.m = m
        self.surf = 2*np.pi*r**2 # Surface of half a sphere, hit by the wind
        self.p = np.array([0.0,0.0],dtype=float)
        self.v = np.array([0.0,0.0],dtype=float)
        self.dt = dt
    
    def __continuos_dynamics(self,x,u):
        dpdt = x[2:]
        dvdt = 1/self.m*u
        return np.concatenate([dpdt,dvdt])

    def discrete_dynamics(self,force):
        '''
        Apply RK4 discretization to compute the discrete dynamics.
        '''
        x = np.concatenate([self.p,self.v])
        k1 = self.dt*self.__continuos_dynamics(x,force)
        k2 = self.dt*self.__continuos_dynamics(x+0.5*k1,force)
        k3 = self.dt*self.__continuos_dynamics(x+0.5*k2,force)
        k4 = self.dt*self.__continuos_dynamics(x+k3,force)

        x = x+(k1+2*k2+2*k3+k4)/6
        self.p = x[0:2]
        self.v = x[2:]

    def set_state(self, p, v):
        self.p = p
        self.v = v