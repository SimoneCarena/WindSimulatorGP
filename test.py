import numpy as np
import matplotlib.pyplot as plt

from modules.Quadrotor import Quadrotor
from modules.MPC import MPC
from modules.Trajectory import Trajectory

trajectory = Trajectory('trajectories/lemniscate.mat',1,[2,2])
target_p, target_v = trajectory.trajectory()

# Instantiate the Quadrotor and MPCController
dt = 0.01
x0 = np.zeros(10)
quadrotor = Quadrotor(dt, x0)
control_horizon = 10
mpc = MPC(
    quadrotor.get_ideal_dynamics(),
    control_horizon=control_horizon, 
    dt=dt, 
    Q=3000*np.eye(4), 
    R=0.0001*np.eye(4)
)

# Example usage in a simulation loop
time_steps = 100
states = []
T = []
u_phi = []
u_theta = []
u_psi = []
u_a = []

# for t in range(len(target_p[0,:])):
#     print(t)
#     # Get state
#     x_current = quadrotor.get_state()
#     states.append(x_current)
#     # Get trajectory for horizon
#     idx = min(t+control_horizon,len(target_p[0,:]))
#     state = np.repeat(x_current[:,np.newaxis], control_horizon, axis=1)
#     ref = np.concatenate([
#             target_p[:,t:idx]
#     ])
#     # If the remaining trajectory is < than the control horizon
#     # expand it using the last refence
#     if idx-t<=control_horizon:
#         ref = np.concatenate([
#             ref,
#             np.repeat(ref[:,-1,np.newaxis],control_horizon-(idx-t),axis=1)
#         ],axis=1)
#     u = mpc(state,ref)
#     u = np.array(u).flatten()
#     u_phi.append(u[0])
#     u_theta.append(u[1])
#     u_psi.append(u[2])
#     u_a.append(u[3])
#     # Compute System's Dynamics
#     quadrotor.step(u, np.zeros(3)) 
#     T.append(t*dt)

ref = np.array([0,np.pi/12,0,12])
ref = np.repeat(
        ref[:,np.newaxis],
        control_horizon,
        axis=1
    )

print('Simulating...')
for t in range(time_steps):
    state = quadrotor.get_state()
    states.append(state.copy())
    u = mpc(state,ref)
    u = np.array(u).flatten()
    u_phi.append(u[0])
    u_theta.append(u[1])
    u_psi.append(u[2])
    u_a.append(u[3])
    # Compute System's Dynamics
    quadrotor.step(u, np.zeros(3)) 
    T.append(t*dt)

    print(
        '|{}{}| {:.2f}% ({:.2f}/{:.2f} s)'.format(
            '█'*int(20*(t+1)/time_steps),
            ' '*(20-int(20*(t+1)/time_steps)),
            (t+1)/time_steps*100,
            (t+1)*dt,
            time_steps*dt
        ),
        end='\r'
    )

print('')

xs = []
ys = []
zs = []
vxs = []
vys = []
vzs = []
phis = []
thetas = []
psis = []
thrusts = []
for state in states:
    state = np.array(state).flatten()
    xs.append(state[0])
    ys.append(state[1])
    zs.append(state[2])
    vxs.append(state[3])
    vys.append(state[4])
    vzs.append(state[5])
    phis.append(state[6])
    thetas.append(state[7])
    psis.append(state[8])
    thrusts.append(state[9])

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
# ax.plot(target_p[0,:],target_p[1,:],target_p[2,:],'--',color='orange',label='Reference Trajectory')
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
fig.suptitle(r'$x$-Axis Evolution')
ax[0].plot(T,xs,label='System Position x')
# ax[0].plot(T,target_p[0,:],'--',color='orange',label='Reference Position')
ax[0].legend()
ax[1].plot(T,vxs,label='System Velocity x')
# ax[1].plot(T,target_v[0,:],'--',color='orange',label='Reference Velocity')
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
fig.suptitle(r'$y$-Axis Evolution')
ax[0].plot(T,ys,label='System Position y')
# ax[0].plot(T,target_p[1,:],'--',color='orange',label='Reference Position')
ax[0].legend()
ax[1].plot(T,vys,label='System Velocity y')
# ax[1].plot(T,target_v[1,:],'--',color='orange',label='Reference Velocity')
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
fig.suptitle(r'$z$-Axis Evolution')
ax[0].plot(T,zs,label='System Position z')
# ax[0].plot(T,target_p[2,:],'--',color='orange',label='Reference Position')
ax[0].legend()
ax[1].plot(T,vzs,label='System Velocity z')
# ax[0].plot(T,target_v[2,:],'--',color='orange',label='Reference Velocity')
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
fig.suptitle(r'$\phi$ Evolution')
ax.plot(T,phis,label=r'System Attitue $\phi$')
ax.plot([T[0],T[-1]],[0,0],'--',color='orange',label=r'Reference Attitue $\phi_{ref}$')
ax.set_xlim([0.0,time_steps*dt])
ax.set_xlabel(r'$t$ $[s]$')
ax.set_ylabel(r'Angle $[rad]$')
ax.legend()
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
fig.suptitle(r'$\theta$ Evolution')
ax.plot(T,thetas,label=r'System Attitue $\theta$')
ax.plot([T[0],T[-1]],[np.pi/12,np.pi/12],'--',color='orange',label=r'Reference Attitue $\theta_{ref}$')
ax.set_xlim([0.0,time_steps*dt])
ax.set_xlabel(r'$t$ $[s]$')
ax.set_ylabel(r'Angle $[rad]$')
ax.legend()
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
fig.suptitle(r'$\phi$ Evolution')
ax.plot(T,psis,label=r'System Attitue $\psi$')
ax.plot([T[0],T[-1]],[0,0],'--',color='orange',label=r'Reference Attitue $\psi_{ref}$')
ax.set_xlim([0.0,time_steps*dt])
ax.set_xlabel(r'$t$ $[s]$')
ax.set_ylabel(r'Angle $[rad]$')
ax.legend()
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
fig.suptitle(r'Thrust Evolution')
ax.plot(T,phis,label=r'System Thrust $a$')
ax.plot([T[0],T[-1]],[0,0],'--',color='orange',label=r'Reference Thrust $a_{ref}$')
ax.set_xlim([0.0,time_steps*dt])
ax.set_xlabel(r'$t$ $[s]$')
ax.set_ylabel(r'Angle $[rad]$')
ax.legend()
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

plt.close('all')