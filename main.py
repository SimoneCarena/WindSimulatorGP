import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import json
from Fan import Fan
from System import System
from PID import PID
import argparse

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--save_plots',default='None')

# Parse Arguments
args = parser.parse_args()
## Parse 'save-plots' options
save_plots = 'None'
if args.save_plots == 'imgs':
    save_plots = 'imgs'
elif args.save_plots == 'full':
    save_plots = 'full'


file = open('configs/wind_field.json')
data = json.load(file)

# Parse wind field and simulation data
width = data["width"]
height = data["height"]
duration = data["duration"] # Number of steps taken
dt = data["dt"] # Sampling time
air_density = data["air_density"] 

# Parse fans' data
fans = []
for fan in data["fans"]:
    x0 = float(fan["x0"])
    y0 = float(fan["y0"])
    alpha = np.deg2rad(float(fan["alpha"]))
    theta = float(fan["theta"])
    v0 = float(fan["v0"])
    noise_var = float(fan['noise_var'])

    u0 = np.array([1,0])
    rot_mat = np.array([
        [np.cos(alpha),-np.sin(alpha)],
        [np.sin(alpha),np.cos(alpha)]
    ],dtype=float)
    u0 = rot_mat@u0

    f = Fan(x0,y0,u0[0],u0[1],theta,v0,noise_var)
    fans.append(f)
file.close()

# Parse system data
file = open('configs/mass.json')
data = json.load(file)
m = data["m"]
r = data["r"]
x0 = data["x0"]
y0 = data["y0"]
v0x = data["v0x"]
v0y = data["v0y"]

system = System(m,r,x0,y0,v0x,v0y,dt)
# The controller's parameter were retrieved using MATLAB
pid = PID(
    16.255496588371, # Proportional
    6.40173078542831, # Integral
    9.79714803790873, # Derivative
    dt # Sampling time
)

target = np.array([2.5,2.5]) # Target Position
xs = [] # List of x positions
ys = [] # List of y positions
vxs = [] # List of x velocities
vys = [] # List of y velocities
ctl_forces_x = [] # List of x control forces
ctl_forces_y = [] # List of y control forces
wind_force_x = [] # List of x wind forces
wind_force_y = [] # List of y wind forces
ex = [] # List of x position traking errors
ey = [] # List of y position traking errors

# Simulate the wind in the field
def simulate_wind_field(): 

    total_speed = np.array([0,0],dtype=float)
    for fan in fans:
        speed = fan.generate_wind(system.p[0],system.p[1])
        total_speed+=speed
    
    # Generate control force
    error = target - system.p
    control_force = pid.step(error)
    # Generate wind force
    wind_force = (0.5*air_density*system.surf)*total_speed**2*np.sign(total_speed)
    # Total force
    force = wind_force + control_force
    # Simulate Dynamics
    system.discrete_dynamics(force)
    xs.append(system.p[0])
    ys.append(system.p[1])
    vxs.append(system.v[0])
    vys.append(system.v[1])
    ctl_forces_x.append(control_force[0])
    ctl_forces_y.append(control_force[0])
    ex.append(error[0])
    ey.append(error[1])
    wind_force_x.append(wind_force[0])
    wind_force_y.append(wind_force[1])


for t in range(duration):
    simulate_wind_field()
T = [t*dt for t in range(duration)]

fig, ax = plt.subplots(1,2)
ax[0].plot(T,xs,label='Object Position')
ax[0].plot([0,T[-1]],[target[0],target[0]],label='Reference Position')
ax[0].title.set_text(r'Position ($x$)')
ax[0].legend()
ax[0].set_xlabel(r'$t$ $[s]$')
ax[0].set_ylabel(r'$x$ $[m]$')
ax[1].plot(T,ex)
ax[1].title.set_text(r'Traking error ($e_x$)')
ax[1].set_xlabel(r'$t$ $[s]$')
ax[1].set_ylabel(r'$e_x$ $[m]$')
fig.set_size_inches(16,9)
if save_plots != 'None':
    plt.savefig('imgs/x-position.png',dpi=300)
plt.show()

fig, ax = plt.subplots(1,2)
ax[0].plot(T,ys,label='Object Position')
ax[0].plot([0,T[-1]],[target[1],target[1]],label='Reference Position')
ax[0].title.set_text(r'Position ($y$)')
ax[0].legend()
ax[0].set_xlabel(r'$t$ $[s]$')
ax[0].set_ylabel(r'$y$ $[m]$')
ax[1].plot(T,ey)
ax[1].title.set_text(r'Traking error ($e_y$)')
ax[1].set_xlabel(r'$t$ $[s]$')
ax[1].set_ylabel(r'$e_y$ $[m]$')
fig.set_size_inches(16,9)
if save_plots != 'None':
    plt.savefig('imgs/y-position.png',dpi=300)
plt.show()

fig, ax = plt.subplots(1,2)
ax[0].plot(T,vxs)
ax[0].set_xlabel(r'$t$ $[s]$')
ax[0].set_ylabel(r'$V_x$ $[m/s]$')
ax[0].title.set_text(r'Velocity ($V_x$)')
ax[1].plot(T,vys)
ax[1].set_xlabel(r'$t$ $[s]$')
ax[1].set_ylabel(r'$V_y$ $[m/s]$')
ax[1].title.set_text(r'Velocity ($V_y$)')
fig.set_size_inches(16,9)
if save_plots != 'None':
    plt.savefig('imgs/velocity.png',dpi=300)
plt.show()

fig, ax = plt.subplots(1,2)
ax[0].plot(T,ctl_forces_x)
ax[0].set_xlabel(r'$t$ $[s]$')
ax[0].set_ylabel(r'$u_x$ $[N]$')
ax[0].title.set_text(r'Control Force ($u_x$)')
ax[1].plot(T,ctl_forces_y)
ax[1].set_xlabel(r'$t$ $[s]$')
ax[1].set_ylabel(r'$u_y$ $[N]$')
ax[1].title.set_text(r'Control Force ($u_y$)')
fig.set_size_inches(16,9)
if save_plots != 'None':
    plt.savefig('imgs/control-force.png',dpi=300)
plt.show()

fig, ax = plt.subplots(1,2)
ax[0].plot(T,wind_force_x)
ax[0].set_xlabel(r'$t$ $[s]$')
ax[0].set_ylabel(r'$F_{wx}$ $[N]$')
ax[0].title.set_text(r'Wind Force ($F_{wx}$)')
ax[1].plot(T,wind_force_y)
ax[1].set_xlabel(r'$t$ $[s]$')
ax[1].set_ylabel(r'$F_{wy}$ $[N]$')
ax[1].title.set_text(r'Wind Force ($F_{wy}$)')
fig.set_size_inches(16,9)
if save_plots != 'None':
    plt.savefig('imgs/wind-force.png',dpi=300)
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(16,9)
def animation_function(t):
    ax.clear()
    ax.set_xlim([0,width])
    ax.set_ylim([0,height])
    ax.plot(xs[t],ys[t],'ko',markersize=5,label='t={0:.2f} s'.format(t*dt)) # Object Moving
    ax.plot(target[0],target[1],'go',markersize=7,label='Target Distance=[{0:.2f},{0:.2f}] m'.format(ex[t],ey[t])) # Traget Location
    ax.quiver(xs[t],ys[t],wind_force_x[t],wind_force_y[t],scale=20,width=0.003,color='r',label='Wind Force=[{0:.2f},{0:.2f}] N'.format(wind_force_x[t],wind_force_y[t])) # Wind Force
    ax.quiver(xs[t],ys[t],ctl_forces_x[t],ctl_forces_y[t],scale=20,width=0.003,color='b',label='Control Force=[{0:.2f},{0:.2f}] N'.format(ctl_forces_x[t],ctl_forces_y[t])) # Control Force
    # Plot fans
    for fan in fans:
        ax.quiver(fan.p0[0],fan.p0[1],fan.u0[0],fan.u0[1],scale=10,color='k')
    ax.legend()

anim = animation.FuncAnimation(fig,animation_function,frames=duration,interval=10,repeat=False)
if save_plots == 'full':
    anim.save('imgs/simulation.gif', fps=30)
plt.show()