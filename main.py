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

# Simulate the wind in the field
def simulate_wind_field(t):
    ax.clear()
    for x in np.linspace(0,width,resolution):
        for y in np.linspace(0,height,resolution):
            # Total wind speed in each position
            total_speed = np.array([0,0],dtype=float)
            for fan in fans:
                speed = fan.generate_wind(x,y)
                total_speed+=speed
            if total_speed[0]==0 and total_speed[1]==0:
                continue
            ax.quiver(x,y,total_speed[0],total_speed[1],width=0.0035,color='r',scale=80)

anim = animation.FuncAnimation(fig,simulate_wind_field,frames=10,repeat=False)
plt.show()