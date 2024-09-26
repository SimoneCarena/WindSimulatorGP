import numpy as np

Q = 100*np.eye(6)
R = 0.1*np.eye(4)

nx = 10
ny = 6
nu = 4

W = np.block([
    [Q,np.zeros((ny,nu))],
    [np.zeros((nu,ny)),R]
])
Vx = np.diag([1,1,1,1,1,1,0,0,0,0])
x = np.arange(1,11)
Vu = np.block([
    [np.zeros((ny,nu))],
    [np.eye(nu)]
])
u = np.arange(-4,0)

print(Vx@x,'\n')
print(Vu@u,'\n')
print(Vx@x+Vu@u,'\n')
print(
    (Vx@x+Vu@u).T@W,
    '\n'
)