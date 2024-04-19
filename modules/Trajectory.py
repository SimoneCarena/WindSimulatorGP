from scipy.io import loadmat

class Trajectory:
    '''
    Iterator used to describe the trajectory at each time step
    '''
    def __init__(self, path):
        mat = loadmat(path) # Load Trajectory File
        self.q = mat.get('q')
        self.current = 0 # Iterator counter
        self.size = len(self.q[0,:])

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.size:
            val = self.q[:,self.current]
            self.current += 1
            return val
        raise StopIteration

    def __len__(self):
        return self.size
    
    def trajectory(self):
        '''
        Returns the whole trajectory
        '''
        return self.q.copy()