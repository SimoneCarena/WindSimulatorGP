import numpy as np

from scipy.io import loadmat

class Trajectory:
    '''
    Iterator used to describe the trajectory at each time step
    '''
    def __init__(self, path, laps, extend: tuple[float, float] = None):
        mat = loadmat(path)
        data = mat.get('data')
        p = data[:2,:]
        v = data[2:,:]
        if extend is not None:
            pz = extend[0]
            p = np.concatenate([p,pz*np.ones((1,data.shape[1]))])
            vz = extend[1]
            v = np.concatenate([v,vz*np.ones((1,data.shape[1]))])
        self.p = p
        self.v = v
        for _ in range(1,laps):
            self.p = np.concatenate([self.p,p],axis=1)
            self.v = np.concatenate([self.v,v],axis=1)
        self.current = 0 # Iterator counter
        self.size = len(data[0,:])*laps

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