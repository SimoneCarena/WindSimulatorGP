import numpy as np
import matplotlib.pyplot as plt

from modules.DynamicallyFeasibleTrajectory import Trajectory
from modules.Quadrotor import Quadrotor

quadrotor = Quadrotor(0.1, np.zeros(10))
N = 10

trajectory = Trajectory(
    quadrotor.get_ideal_dynamics(),
    1,
    N,
    10,
    4
)

x0 = np.zeros(10)
s = np.linspace(0,1,N).reshape(N,1)
p0 = np.zeros((3,1))
pf = 2*np.ones((3,1))
ref = p0@(1-s).T + pf@s.T

x_opt, u_opt, t_opt = trajectory(x0,ref)
print(f"Optimal Execution Time = {t_opt}")

fig = plt.figure()
ax = plt.axes(projection='3d')
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
fig.suptitle('System Trajectory')
ax.plot(
    x_opt[0,:],
    x_opt[1,:],
    x_opt[2,:],
)

fig, ax = plt.subplots()
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
fig.suptitle('Control Variable (Attitude)')
ax.plot(u_opt[0,:],label=r'$\phi^c$')
ax.plot(u_opt[1,:],label=r'$\theta^c$')
ax.plot(u_opt[2,:],label=r'$\psi^c$')
ax.set_xlabel(r'steps')
ax.set_ylabel(r'angle $[rad]$')
ax.legend()

fig, ax = plt.subplots()
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
fig.suptitle('Control Variable (Thrust)')
ax.plot(u_opt[3,:],label=r'$a^c$')
ax.set_xlabel(r'steps')
ax.set_ylabel(r'thrust $[m/s^2]$')
ax.legend()

fig, ax = plt.subplots(3,1)
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
fig.suptitle('System Evolution')
ax[0].plot(x_opt[0,:],label=r'$x$')
ax[0].plot([0,N],[2,2],'--',color='orange',label=r'$x_{ref}$')
ax[1].plot(x_opt[1,:],label=r'$y$')
ax[1].plot([0,N],[2,2],'--',color='orange',label=r'$y_{ref}$')
ax[2].plot(x_opt[2,:],label=r'$z$')
ax[2].plot([0,N],[2,2],'--',color='orange',label=r'$z_{ref}$')
ax[0].set_xlabel(r'steps')
ax[0].set_ylabel(r'x $[m]$')
ax[1].set_xlabel(r'steps')
ax[1].set_ylabel(r'y $[m]$')
ax[2].set_xlabel(r'steps')
ax[2].set_ylabel(r'z $[m]$')
ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.show()

