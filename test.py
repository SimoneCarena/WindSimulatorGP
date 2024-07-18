import numpy as np
import matplotlib.pyplot as plt

from modules.Quadrotor import Quadrotor
from modules.MPC import MPC

# Instantiate the Quadrotor and MPCController
dt = 0.1
x0 = np.zeros(10)
quadrotor = Quadrotor(dt, x0)
mpc = MPC(
    quadrotor.ideal_dynamics,
    control_horizon=5, 
    dt=dt, 
    Q=100*np.diag([5,5,5,1,1,1]), 
    R=np.eye(4)
)

# Example usage in a simulation loop
time_steps = 200
states = []
T = []
u_phi = []
u_theta = []
u_psi = []
u_a = []

for t in range(time_steps):
    x_current = quadrotor.get_state()
    states.append(x_current)
    u = mpc(x_current,np.array([2,2,2,0,0,0]))
    u_phi.append(np.array(u[0]).item())
    u_theta.append(np.array(u[1]).item())
    u_psi.append(np.array(u[2]).item())
    u_a.append(np.array(u[3]).item())
    quadrotor.step(u, np.zeros(3))  # Assuming no wind disturbance for ideal dynamics
    T.append(t*dt)

xs = []
ys = []
zs = []
vxs = []
vys = []
vzs = []
for state in states:
    xs.append(np.array(state[0]).item())
    ys.append(np.array(state[1]).item())
    zs.append(np.array(state[2]).item())
    vxs.append(np.array(state[3]).item())
    vys.append(np.array(state[4]).item())
    vzs.append(np.array(state[5]).item())

xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)

max_z = np.max(zs)

fig = plt.figure()
ax = plt.axes(projection='3d')
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
fig.suptitle('System Trajectory')
ax.plot(xs,ys,zs,'-',color='c',label='System Trajectory')
ax.plot(xs[0],ys[0],zs[0],'bo',label='Starting Position')
ax.plot(xs[-1],ys[-1],zs[-1],'ro',label='End Position')
ax.set_xlabel(r'$x$ $[m]$')
ax.set_ylabel(r'$y$ $[m]$')
ax.set_zlabel(r'$z$ $[m]$')
ax.legend()
plt.show()

fig, ax = plt.subplots(1,2)
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
ax[0].plot(T,xs,label='System Position x')
ax[0].plot([T[0],T[-1]],[2,2],label='Reference Position')
ax[0].legend()
ax[1].plot(T,vxs,label='System Velocity x')
ax[1].plot([T[0],T[-1]],[0,0],label='Reference Position')
ax[1].legend()
ax[0].set_xlabel(r'$t$ $[s]$')
ax[0].set_ylabel(r'$x$ $[m]$')
ax[1].set_xlabel(r'$t$ $[s]$')
ax[1].set_ylabel(r'$v_x$ $[m/s]$')
ax[0].set_xlim([0.0,time_steps*dt])
ax[1].set_xlim([0.0,time_steps*dt])
plt.show()

fig, ax = plt.subplots(1,2)
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
ax[0].plot(T,ys,label='System Position y')
ax[0].plot([T[0],T[-1]],[2,2],label='Reference Position')
ax[0].legend()
ax[1].plot(T,vys,label='System Velocity y')
ax[1].plot([T[0],T[-1]],[0,0],label='Reference Position')
ax[1].legend()
ax[0].set_xlabel(r'$t$ $[s]$')
ax[0].set_ylabel(r'$y$ $[m]$')
ax[1].set_xlabel(r'$t$ $[s]$')
ax[1].set_ylabel(r'$v_y$ $[m/s]$')
ax[0].set_xlim([0.0,time_steps*dt])
ax[1].set_xlim([0.0,time_steps*dt])
plt.show()

fig, ax = plt.subplots(1,2)
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
ax[0].plot(T,xs,label='System Position z')
ax[0].plot([T[0],T[-1]],[2,2],label='Reference Position')
ax[0].legend()
ax[1].plot(T,vxs,label='System Velocity z')
ax[1].plot([T[0],T[-1]],[0,0],label='Reference Position')
ax[1].legend()
ax[0].set_xlabel(r'$t$ $[s]$')
ax[0].set_ylabel(r'$z$ $[m]$')
ax[1].set_xlabel(r'$t$ $[s]$')
ax[1].set_ylabel(r'$v_z$ $[m/s]$')
ax[0].set_xlim([0.0,time_steps*dt])
ax[1].set_xlim([0.0,time_steps*dt])
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
fig.suptitle('Attitude Command')
ax.plot(T,u_phi,label=r'$\phi^{c}$')
ax.plot(T,u_theta,label=r'$\theta^{c}$')
ax.plot(T,u_psi,label=r'$\psi^{c}$')
ax.plot([T[0],T[-1]],[-np.pi/6,-np.pi/6],'k--',label='Bounds')
ax.plot([T[0],T[-1]],[np.pi/6,np.pi/6],'k--')
ax.set_xlabel(r'$t$ $[s]$')
ax.set_ylabel(r'angle $[rad]$')
ax.set_xlim([0.0,time_steps*dt])
ax.legend()
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
fig.suptitle('Thrust Command')
ax.plot(T,u_a,label=r'$a^{c}$')
ax.plot([T[0],T[-1]],[5,5],'k--',label='Bounds')
ax.plot([T[0],T[-1]],[15,15],'k--')
ax.set_xlabel(r'$t$ $[s]$')
ax.set_ylabel(r'thrust $[m/s^2]$')
ax.set_xlim([0.0,time_steps*dt])
ax.legend()
plt.show()