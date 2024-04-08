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

x = 1
y = 2
m = 0.2 
v = np.array([0,0],dtype=float)

# Simulate the wind in the field
def simulate_wind_field(t): # t = 1ms
    # Total wind speed in each position
    global x
    global y
    global v
    # print(x,y,v)
    ax.clear()
    ax.set_xlim([0,width])
    ax.set_ylim([0,height])
    total_speed = np.array([0,0],dtype=float)
    for fan in fans:
        ax.quiver(fan.p0[0],fan.p0[1],fan.u0[0],fan.u0[1],color='k',scale=10)
        ax.plot(fan.p0[0],fan.p0[1],'ko',markersize=5)
        speed = fan.generate_wind(x,y)
        total_speed+=speed
    ax.quiver(x,y,total_speed[0],total_speed[1],color='r',scale=20)
    ax.plot(x,y,'ko',markersize=5)
    # Simple Physics simulation
    # p = 1/2at^2+v0t+p0
    t = t/1000
    F = 3.77*10**(-4)*total_speed**2*np.sign(total_speed) # wind force
    a = F/m
    x = 0.5*a[0]*t**2+v[0]*t+x
    y = 0.5*a[1]*t**2+v[1]*t+y
    v = a*t+v

anim = animation.FuncAnimation(fig,simulate_wind_field,frames=resolution,interval=10)
plt.show()