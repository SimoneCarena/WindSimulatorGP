import numpy as np

from modules.Obstacle import Obstacle

class Fan:
    '''
    Class describing the fans generating the wind. The parameters
    are the position of the fan p0=(x0,y0), the orientation of the fan
    u0=(ux,uy), the spread of the fan theta and the initial wind speed.
    A noise variance can also be specified, and will add additive noise
    to the generated wind from the fan. The resulting covariance matrix will be
    noise_var*eye(2,2)
    '''
    def __init__(self,x0,y0,ux,uy,width,noise_var=0.0,wind_funct=None,obstacles=[]):
        self.p0 = np.array([x0,y0],dtype=float)
        self.u0 = np.array([ux,uy],dtype=float)
        self.noise_mean = 0.0
        self.noise_cov = noise_var
        self.wind_funct = wind_funct
        self.width = width
        self.obstacles = obstacles

    def generate_wind(self,x,y,t):
        '''
        Method used to generate the wind at a certain location (x,y).
        The function returns the wind speed at the specified coordinates
        '''
        for obstacle in self.obstacles:
            if obstacle.is_inside(x,y):
                return np.array([0.0,0.0],dtype=float)
        p = np.array([x-self.p0[0],y-self.p0[1]])
        # Compute wind source and add noise
        v0 = self.wind_funct(t)*self.u0 + np.random.normal(self.noise_mean,self.noise_cov)
        # Linearly scale along u-axis
        ## Project the point p=(x,y) along u
        ## u is already norm-1, so no normalization is needed
        du_par = np.abs(np.dot(self.u0,p))
        # Linearly scale along the direction perpendicular to u
        # but having scale = 0 if the distance anlong such axis is
        # greater than half the width of the fan
        ## Compute u^T (just a 90 rotation of u)
        ut = np.array([[0.0, -1.0],[1.0,0.0]],dtype=float)@self.u0
        ## Project p onto u^T
        du_perp = np.abs(np.dot(ut,p))
        w = 0.02
        # Compute the speed
        speed = v0/2*(np.tanh((du_perp+self.width/2)/(w*(du_par+1)))-np.tanh((du_perp-self.width/2)/(w*(du_par+1))))/(du_par+1)
        
        return speed
        