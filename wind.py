import matplotlib.pyplot as plt
import numpy as np

max_scale = 10
initial_headwidth = 3
initial_headlenght = 4
theta = np.deg2rad(30)

plt.xlim([0, 4])
plt.ylim([0, 4])

x0 = 4
y0 = 2
direction = np.array([-1,0])

plt.plot(x0,y0,"ko",markersize=8)
plt.quiver(x0,y0,direction[0],direction[1],color='k',scale=max_scale)

for x in np.linspace(0,4,20):
    for y in np.linspace(0,4,30):
        d = (x-x0)**2+(y-y0)**2
        u = np.array([(x-x0),(y-y0)])
        u = u/np.linalg.norm(u)
        # Scale the speed with ditance
        distance_scale = 1+d
        # Check if the point is inside the cone
        p0 = np.sqrt(d)*direction+np.array([x0,y0])
        alpha = 2*np.arcsin(np.sqrt((x-p0[0])**2+(y-p0[1])**2)/(2*np.sqrt(d)))
        if alpha > theta:
            continue
        scale = max_scale*distance_scale #This scale is to draw the lines only
        plt.quiver(x,y,u[0],u[1],scale=scale,width=0.0035,color='r')

plt.show()