class Obstacle:
    '''
    Cylyndrical obstacle at position (x,y) and of radius r
    '''
    def __init__(self,x,y,r):
        self.x = x
        self.y = y
        self.r = r
        
    def is_inside(self,x,y):
        if (x-self.x)**2+(y-self.y)**2<=self.r**2:
            return True
        else:
            return False
