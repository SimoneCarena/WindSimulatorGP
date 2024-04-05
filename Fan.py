import numpy as np

class Fan:
    '''
    Class describing the fans generating the wind. The parameters
    are the position of the fan p0=(x0,y0), the orientation of the fan
    u0=(ux,uy), the spread of the fan theta and the initial wind speed
    '''
    def __init__(self,x0,y0,ux,uy,theta,v0):
        self.p0 = np.array([x0,y0])
        self.theta = np.deg2rad(theta)
        self.u0 = np.array([ux,uy])
        self.v0 = v0

    def generate_wind(self,x,y):
        '''
        Method used to generate the wind at a certain location (x,y).
        The function returns the wind speed at the specified coordinates
        '''
        d = (x-self.p0[0])**2+(y-self.p0[1])**2
        u = np.array([(x-self.p0[0]),(y-self.p0[1])])
        u = u/np.linalg.norm(u)
        # Scale the speed with ditance
        distance_scale = 1+d
        # Check if the point is inside the cone
        p = np.sqrt(d)*self.u0+np.array([self.p0[0],self.p0[1]])
        # The angle is computed using traingles properties
        alpha = 2*np.arcsin(np.sqrt((x-p[0])**2+(y-p[1])**2)/(2*np.sqrt(d)))
        if alpha < self.theta:
            speed = self.v0/distance_scale*np.copy(self.u0)
        else:
            speed = np.array([0,0])
        return speed
        