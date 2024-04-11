from scipy.io import loadmat
import numpy as np

class Trajectory:
    def __init__(self, file):
        mat = loadmat(file) # Load Trajectory File
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