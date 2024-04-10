import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import json
from Fan import Fan
from System import System
from Control import Control

#fig,ax = plt.subplots()

file = open('./wind_field.json')
data = json.load(file)

# Parse wind field data
width = data["width"]
height = data["height"]
duration = data["duration"]

# Parse fans data
fans = []
for fan in data["fans"]:
    x0 = float(fan["x0"])
    y0 = float(fan["y0"])
    alpha = np.deg2rad(float(fan["alpha"]))
    theta = float(fan["theta"])
    v0 = float(fan["v0"])
    noise_var = float(fan['noise_var'])

    u0 = np.array([1,0,0])
    rot_mat = np.array([
        [np.cos(alpha),-np.sin(alpha),0],
        [np.sin(alpha),np.cos(alpha),0],
        [0,0,1]
    ],dtype=float)
    u0 = rot_mat@u0
    ux = u0[0]
    uy = u0[1]

    f = Fan(x0,y0,ux,uy,theta,v0,noise_var)
    fans.append(f)


air_density = 1.2
system = System(1,0.1,2,1,0,0,0.01)
controller = Control(
    2.09666862414187, # Proportional
    0.347675754694788, # Integral
    3.12100832597623,
    0.01
)
target = np.array([3,2])
xs = []
ys = []
vxs = []
vys = []
ctl_forces_x = []
ctl_forces_y = []
wind_force_x = []
wind_force_y = []
ex = []
ey = []

# Simulate the wind in the field
def simulate_wind_field(): 

    total_speed = np.array([0,0],dtype=float)
    for fan in fans:
        speed = fan.generate_wind(system.p[0],system.p[1])
        total_speed+=speed
    
    # Generate control output
    error = target - system.p
    control_force = controller.step(error)
    # Generate wind Forces
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
T = [t*0.001 for t in range(duration)]

fig, ax = plt.subplots(1,2)
ax[0].plot(T,xs,label='Object Position')
ax[0].plot([0,T[-1]],[target[0],target[0]],label='Reference Position')
ax[0].title.set_text(r'Position ($x$)')
ax[0].legend()
ax[0].set_xlabel(r'$t$ $[s]$')
ax[0].set_ylabel(r'$x$ $[m]$')
ax[1].plot(T,ex)
ax[1].title.set_text(r'$x$ Traking error')
ax[1].set_xlabel(r'$t$ $[s]$')
ax[1].set_ylabel(r'$e_x$ $[m]$')
plt.show()

fig, ax = plt.subplots(1,2)
ax[0].plot(T,ys,label='Object Position')
ax[0].plot([0,T[-1]],[target[1],target[1]],label='Reference Position')
ax[0].title.set_text(r'Position ($y$)')
ax[0].legend()
ax[0].set_xlabel(r'$t$ $[s]$')
ax[0].set_ylabel(r'$y$ $[m]$')
ax[1].plot(T,ey)
ax[1].title.set_text(r'$y$ Traking error')
ax[1].set_xlabel(r'$t$ $[s]$')
ax[1].set_ylabel(r'$e_y$ $[m]$')
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
plt.show()

fig, ax = plt.subplots()
def animation_function(t):
    ax.clear()
    ax.set_xlim([0,width])
    ax.set_ylim([0,height])
    ax.plot(target[0],target[1],'go',markersize=7) # Traget Location
    ax.plot(xs[t],ys[t],'ko',markersize=5,label='t={0:.2f} s'.format(t*0.01)) # Object Moving
    ax.quiver(xs[t],ys[t],wind_force_x[t],wind_force_y[t],scale=10,width=0.005,headlength=4,color='r',label='Wind Force=[{0:.2f},{0:.2f}] N'.format(wind_force_x[t],wind_force_y[t])) # Wind Force
    ax.quiver(xs[t],ys[t],ctl_forces_x[t],ctl_forces_y[t],scale=10,width=0.005,headlength=4,color='b',label='Control Force=[{0:.2f},{0:.2f}] N'.format(ctl_forces_x[t],ctl_forces_y[t])) # Control Force
    ax.legend()

anim = animation.FuncAnimation(fig,animation_function,frames=duration,interval=10,repeat=False)
plt.show()