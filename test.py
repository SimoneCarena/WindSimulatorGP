import numpy as np
import matplotlib.pyplot as plt

from modules.Quadrotor import Quadrotor
from modules.MPC import MPC
from modules.Trajectory import Trajectory

trajectory = Trajectory('trajectories/lemniscate.mat',1,[2,0])
target_p, target_v = trajectory.trajectory()

# Instantiate the Quadrotor and MPCController
dt = 0.001
control_frequency = 100
x0 = np.array([
    target_p[0,0], target_p[1,0], target_p[2,0],
    target_v[0,0], target_v[1,0], target_v[2,0],
    0,0,0,0
])
quadrotor = Quadrotor(dt, x0)
control_horizon = 10
mpc = MPC(
    quadrotor.get_ideal_dynamics(),
    control_horizon=control_horizon, 
    dt=dt*control_frequency, 
    Q=3000*np.eye(6), 
    R=0.0001*np.eye(4)
)

states = []
T = []
u_phi = []
u_theta = []
u_psi = []
u_a = []
size = len(target_p[0,:])

u = np.zeros(4)
print('Simulating...')
for t in range(len(target_p[0,:])):
    print(
        '|{}{}| {:.2f}% ({:.2f}/{:.2f} s)'.format(
            '█'*int(20*(t+1)/len(target_p[0,:])),
            ' '*(20-int(20*(t+1)/len(target_p[0,:]))),
            (t+1)/len(target_p[0,:])*100,
            (t+1)*dt,
            len(target_p[0,:])*dt
        ),
        end='\r'
    )
    # Get state
    x_current = quadrotor.get_state()
    states.append(x_current)
    if t%control_frequency == 0:
        idx = min(t+control_horizon*control_frequency,len(target_p[0,:]))
        state = x_current
        ref = np.concatenate([
                target_p[:,t:idx:control_frequency],
                target_v[:,t:idx:control_frequency]
        ],axis=0)
        # If the remaining trajectory is < than the control horizon
        # expand it using the last refence
        if (idx-t)//control_frequency < control_horizon:
            ref = np.concatenate([
                ref,
                np.repeat(ref[:,-1,np.newaxis],control_horizon-(idx-t)//control_frequency,axis=1)
            ],axis=1)
        u = mpc(state,ref)
        u = np.array(u).flatten()
    u_phi.append(u[0])
    u_theta.append(u[1])
    u_psi.append(u[2])
    u_a.append(u[3])
    # Compute System's Dynamics
    quadrotor.step(u, np.zeros(3)) 
    T.append(t*dt)

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
ax.plot(target_p[0,:],target_p[1,:],target_p[2,:],'--',color='orange',label='Reference Trajectory')
ax.plot(xs[0],ys[0],zs[0],'bo',label='Starting Position')
ax.plot(xs[-1],ys[-1],zs[-1],'ro',label='End Position')
ax.set_xlabel(r'$x$ $[m]$')
ax.set_ylabel(r'$y$ $[m]$')
ax.set_zlabel(r'$z$ $[m]$')
ax.set_xlim([0,4])
ax.set_ylim([0,4])
ax.set_zlim([0,4])
ax.legend()
plt.show()

fig, ax = plt.subplots(1,2)
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
fig.suptitle(r'$x$-Axis Evolution')
ax[0].plot(T,xs,label='System Position x')
ax[0].plot(T,target_p[0,:],'--',color='orange',label='Reference Position')
ax[0].legend()
ax[1].plot(T,vxs,label='System Velocity x')
ax[1].plot(T,target_v[0,:],'--',color='orange',label='Reference Velocity')
ax[1].legend()
ax[0].set_xlabel(r'$t$ $[s]$')
ax[0].set_ylabel(r'$x$ $[m]$')
ax[1].set_xlabel(r'$t$ $[s]$')
ax[1].set_ylabel(r'$v_x$ $[m/s]$')
ax[0].set_xlim([0.0,size*dt])
ax[1].set_xlim([0.0,size*dt])
plt.show()

fig, ax = plt.subplots(1,2)
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
fig.suptitle(r'$y$-Axis Evolution')
ax[0].plot(T,ys,label='System Position y')
ax[0].plot(T,target_p[1,:],'--',color='orange',label='Reference Position')
ax[0].legend()
ax[1].plot(T,vys,label='System Velocity y')
ax[1].plot(T,target_v[1,:],'--',color='orange',label='Reference Velocity')
ax[1].legend()
ax[0].set_xlabel(r'$t$ $[s]$')
ax[0].set_ylabel(r'$y$ $[m]$')
ax[1].set_xlabel(r'$t$ $[s]$')
ax[1].set_ylabel(r'$v_y$ $[m/s]$')
ax[0].set_xlim([0.0,size*dt])
ax[1].set_xlim([0.0,size*dt])
plt.show()

fig, ax = plt.subplots(1,2)
fig.set_size_inches(16,9)
fig.tight_layout(pad=5)
fig.suptitle(r'$z$-Axis Evolution')
ax[0].plot(T,zs,label='System Position z')
ax[0].plot(T,target_p[2,:],'--',color='orange',label='Reference Position')
ax[0].legend()
ax[1].plot(T,vzs,label='System Velocity z')
ax[1].plot(T,target_v[2,:],'--',color='orange',label='Reference Velocity')
ax[1].legend()
ax[0].set_xlabel(r'$t$ $[s]$')
ax[0].set_ylabel(r'$z$ $[m]$')
ax[1].set_xlabel(r'$t$ $[s]$')
ax[1].set_ylabel(r'$v_z$ $[m/s]$')
ax[0].set_xlim([0.0,size*dt])
ax[1].set_xlim([0.0,size*dt])
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
ax.set_xlim([0.0,size*dt])
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
ax.set_xlim([0.0,size*dt])
ax.legend()
plt.show()

plt.close('all')