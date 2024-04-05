import matplotlib.pyplot as plt
import numpy as np
import json
from Fan import Fan

max_scale=10
initial_headwidth = 3
initial_headlenght = 4
theta = np.deg2rad(30)

plt.xlim([0, 4])
plt.ylim([0, 4])

x0 = 2
y0 = 4
direction = np.array([np.sqrt(2)/2,-np.sqrt(2)/2])

plt.plot(x0,y0,"ko",markersize=8)
plt.quiver(x0,y0,direction[0],direction[1],color='k',scale=max_scale)

fan = Fan(x0,y0,direction[0],direction[1],theta,5)

for x in np.linspace(0,4,20):
    for y in np.linspace(0,4,20):
        speed, scale = fan.generate_wind(x,y)
        if speed[0]==0 and speed[1]==0:
            continue
        speed = speed/np.linalg.norm(speed)
        plt.quiver(x,y,speed[0],speed[1],width=0.0035,color='r',scale=max_scale*scale)

plt.show()