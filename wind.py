import matplotlib.pyplot as plt
import numpy as np

max_scale = 10
initial_headwidth = 3
initial_headlenght = 4
theta = np.deg2rad(10)

plt.xlim([0, 4])
plt.ylim([0, 4])

x0 = 2
y0 = 0
p0 = np.array([x0,y0])
direction = np.array([np.sqrt(2),np.sqrt(2)])

plt.plot(x0,y0,"ko",markersize=8)
plt.quiver(x0,y0,direction[0],direction[1],color='k',scale=max_scale)

for x in np.linspace(0,4,20):
    for y in np.linspace(0,4,30):
        #check is point is within the cone
        m1 = np.tan(np.pi/2-theta/2)
        m2 = np.tan(np.pi/2+theta/2)

        d = (x-x0)**2+(y-y0)**2
        u = np.array([(x-x0),(y-y0)])
        u = u/np.linalg.norm(u)
        scale = max_scale*(1+d)
        plt.quiver(x,y,u[0],u[1],scale=scale,color='r')

plt.show()