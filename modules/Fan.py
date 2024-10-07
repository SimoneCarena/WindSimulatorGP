import numpy as np


class RealFan:
    '''
    Class describing a real set of fans generated the wind. The data is assumed to be stored
    in the constructor parameters. The size of the wind field, as well as the resolution
    of the grid are necessary to compute the indices for when generating the wind.
    Currently only time-invariant wind fields are supported
    '''
    def __init__(self, mean_map, var_map, width, height, resolution):
        self.__mean_map = mean_map
        self.__var_map = var_map
        self.__width = width
        self.__height = height
        self.__resolution = resolution

    def generate_wind(self,x,y,t):
        # Get indeces for the grid
        idx_x = int(x*self.__resolution//self.__width)
        idx_y = int(y*self.__resolution//self.__height)

        # Draw the speed from a multivariate normal with certain speed and covariance
        speed = np.random.multivariate_normal(
            mean = np.array([
                3*self.__mean_map[0,self.__resolution-1-idx_y,idx_x], # x mean component
                3*self.__mean_map[1,self.__resolution-1-idx_y,idx_x]  # y mean component
            ]),
            cov = np.diag([
                # self.__var_map[0,self.__resolution-1-idx_y,idx_x], # x var component
                # self.__var_map[1,self.__resolution-1-idx_y,idx_x]  # y var component
                0,0
            ])
        )
        
        return speed

class SimulatedFan:
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
        