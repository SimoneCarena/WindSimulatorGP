import numpy as np

class Fan:
    '''
    Class describing the fans generating the wind. The parameters
    are the position of the fan p0=(x0,y0), the orientation of the fan
    u0=(ux,uy), the spread of the fan theta and the initial wind speed.
    A noise variance can also be specified, and will add additive noise
    to the generated wind from the fan. The resulting covariance matrix will be
    noise_var*eye(2,2)
    '''
    def __init__(self,x0,y0,ux,uy,theta,v0,noise_var=0.0):
        self.p0 = np.array([x0,y0],dtype=float)
        self.u0 = np.array([ux,uy],dtype=float)
        self.theta = np.deg2rad(theta)
        self.v0 = float(v0)
        self.noise_mean = 0.0
        self.noise_cov = noise_var

    def generate_wind(self,x,y):
        '''
        Method used to generate the wind at a certain location (x,y).
        The function returns the wind speed at the specified coordinates
        '''
        d = (x-self.p0[0])**2+(y-self.p0[1])**2
        u = np.array([(x-self.p0[0]),(y-self.p0[1])])
        u = u/np.linalg.norm(u)
        # Scale the speed with ditance
        distance_scale = (1+d)**2
        # Check if the point is inside the cone
        p = np.sqrt(d)*self.u0+np.array([self.p0[0],self.p0[1]])
        ## The angle is computed using traingles properties
        alpha = 2*np.arcsin(np.sqrt((x-p[0])**2+(y-p[1])**2)/(2*np.sqrt(d)))
        # Radial Scale
        radial_scale =  (1-2*alpha/self.theta)**2
        # Add noise
        v0 = self.v0 + np.random.normal(self.noise_mean,self.noise_cov)
        if alpha < self.theta/2:
            speed =radial_scale*v0/distance_scale*np.copy(u)
        else:
            speed = np.array([0.0,0.0])
        return speed
        