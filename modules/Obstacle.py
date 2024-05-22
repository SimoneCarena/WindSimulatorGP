import numpy as np

class Obstacle:
    '''
    Cylyndrical obstacle at position (x,y) and of radius r
    '''
    def __init__(self,x,y,r):
        self.x = x
        self.y = y
        self.r = r

    def distance(self,x,y):
        return np.sqrt((x-self.x)**2+(y-self.y)**2)
        
    def is_inside(self,x,y):
        if self.distance(x,y)<=self.r:
            return True
        else:
            return False
