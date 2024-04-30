from scipy.io import loadmat

class Trajectory:
    '''
    Iterator used to describe the trajectory at each time step
    '''
    def __init__(self, path):
        mat = loadmat(path)
        data = mat.get('data')
        self.p = data[:2,:]
        self.v = data[2:,:]
        self.current = 0 # Iterator counter
        self.size = len(data[0,:])

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.size:
            p = self.p[:,self.current]
            v = self.v[:,self.current]
            self.current += 1
            return p,v
        raise StopIteration

    def __len__(self):
        return self.size
    
    def trajectory(self):
        '''
        Returns the whole trajectory
        '''
        return self.p.copy(), self.v.copy()