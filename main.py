import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import json
from Fan import Fan

fig,ax = plt.subplots()

file = open('./wind_field.json')
data = json.load(file)

# Parse wind field data
width = data["width"]
height = data["height"]
resolution = data["resolution"]

# Parse fans data
fans = []
for fan in data["fans"]:
    x0 = float(fan["x0"])
    y0 = float(fan["y0"])
    alpha = np.deg2rad(float(fan["alpha"]))
    theta = float(fan["theta"])
    v0 = float(fan["v0"])

    # Comptute the fan direction given the rotation angle alpha
    if x0==0 and y0!=0:
        ux0 = 1
        uy0 = 0
    elif x0!=0 and y0==0:
        ux0 = 0
        uy0 = 1
    elif x0==width and y0!=height:
        ux0 = -1
        uy0 = 0
    elif x0!=width and y0==height:
        ux0 = 0
        uy0 = -1

    # Rotate the original versor to et the desired direction
    ux = (ux0*np.cos(alpha))-(uy0*np.sin(alpha))
    uy = (ux0*np.sin(alpha))+(uy0*np.cos(alpha))

    f = Fan(x0,y0,ux,uy,theta,v0)
    fans.append(f)

# Setup the wind field
ax.set_xlim([0,width])
ax.set_ylim([0,height])

N = int(resolution/4)
tr1 = [(x,3) for x in np.linspace(1,3,N)]
tr2 = [(3,y) for y in np.linspace(3,1,N)]
tr3 = [(x,1) for x in np.linspace(3,1,N)]
tr4 = [(1,y) for y in np.linspace(1,3,N)]
trajectory = tr1+tr2+tr3+tr4

# Simulate the wind in the field
def simulate_wind_field(t):
    # Total wind speed in each position
    ax.clear()
    ax.set_xlim([0,width])
    ax.set_ylim([0,height])
    x = trajectory[t][0]
    y = trajectory[t][1]
    total_speed = np.array([0,0],dtype=float)
    for fan in fans:
        ax.quiver(fan.p0[0],fan.p0[1],fan.u0[0],fan.u0[1],color='k',scale=10)
        ax.plot(fan.p0[0],fan.p0[1],'ko',markersize=5)
        speed = fan.generate_wind(x,y)
        total_speed+=speed
    ax.quiver(x,y,total_speed[0],total_speed[1],color='r',scale=20)
    ax.plot(x,y,'ko',markersize=5)

anim = animation.FuncAnimation(fig,simulate_wind_field,frames=resolution,interval=10)
plt.show()