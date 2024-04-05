import matplotlib.pyplot as plt
import numpy as np
import json
from Fan import Fan

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
    ux = float(fan["ux"])
    uy = float(fan["uy"])
    theta = float(fan["theta"])
    v0 = float(fan["v0"])

    # Plot the fans
    plt.plot(x0,y0,'ko',markersize=5)
    plt.quiver(x0,y0,ux,uy,scale=10)

    f = Fan(x0,y0,ux,uy,theta,v0)
    fans.append(f)

# Setup the wind field
plt.xlim([0,width])
plt.ylim([0,height])

# Simulate the wind in the field
for x in np.linspace(0,width,resolution):
    for y in np.linspace(0,height,resolution):
        total_speed = np.array([0,0],dtype=float)
        for fan in fans:
            speed = fan.generate_wind(x,y)
            total_speed+=speed
        if total_speed[0]==0 and total_speed[1]==0:
            continue
        plt.quiver(x,y,total_speed[0],total_speed[1],width=0.0035,color='r',scale=80)

plt.show()