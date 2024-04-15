import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import json
from Fan import Fan
from System import System
from PID import PID
import argparse
from Trajectory import Trajectory
import os
from pathlib import Path
from ExactGPModel import ExactGPModel
import gpytorch
import torch
import random

# Simulate the wind in the field
def simulate_wind_field(): 
    for (t,target) in enumerate(trajectory):
        total_speed = np.array([0,0],dtype=float)
        for fan in fans:
            speed = fan.generate_wind(system.p[0],system.p[1])
            total_speed+=speed
        
        # Generate control force
        error = target - system.p
        control_force = pid.step(error)
        # Apply only the control force to the model
        p_pred, v_pred = model.discrete_dynamics(control_force)
        # Generate wind force
        wind_force = (0.5*air_density*system.surf)*total_speed**2*np.sign(total_speed)
        # Total force
        force = wind_force + control_force
        # Simulate Dynamics
        p_true, v_true = system.discrete_dynamics(force)
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

        if t == 0:  
            train_data.append([p_true[0]-p_pred[0],p_true[1]-p_pred[1],v_true[0]-v_pred[0],v_true[1]-v_pred[1]])
        elif t == duration-1:
            train_label_x.append(wind_force[0])
            train_label_y.append(wind_force[1])
        else:
            train_data.append([p_true[0]-p_pred[0],p_true[1]-p_pred[1],v_true[0]-v_pred[0],v_true[1]-v_pred[1]])
            train_label_x.append(wind_force[0])
            train_label_y.append(wind_force[1])

        # Update the model to its true new state
        model.set_state(p_true,v_true)

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--save_plots',default='imgs')
parser.add_argument('--show_plots',default='None')

# Parse Arguments
args = parser.parse_args()
## Parse 'save_plots' options
save_plots = 'imgs'
if args.save_plots == 'None':
    save_plots = 'None'
elif args.save_plots == 'full':
    save_plots = 'full'
## Parse 'show_plots' options
show_plots = 'None'
if args.show_plots == 'imgs':
    show_plots = 'imgs'
elif args.show_plots == 'full':
    show_plots = 'full'

# Wind Field Configuration
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

# Create arrays to train the GP model
train_data = []
train_label_x = []
train_label_y = []

# Run the simulation for the different trajectories
for file in os.listdir('trajectories'):

    # Get Trajectory Data
    path = 'trajectories/'+file
    file_name = Path(file).stem

    # Create Plots arrays
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

    # Generate Trajectory
    trajectory = Trajectory(path)
    tr = trajectory.trajectory() # Full Trajectory

    # Parse system data
    file = open('configs/mass.json')
    data = json.load(file)
    m = data["m"]
    r = data["r"]
    v0x = data["v0x"]
    v0y = data["v0y"]
    x0 = tr[0,0]
    y0 = tr[1,0]

    # Actual system moving in the wind-field
    system = System(m,r,x0,y0,v0x,v0y,dt)
    # Model of the system, used to predict next postion and velocity
    # to train the GP. The model predicts the next state without
    # considering the effect of the wind
    model = System(m,r,x0,y0,v0x,v0y,dt)
    # The controller's parameter were retrieved using MATLAB
    pid = PID(
        16.255496588371, # Proportional
        6.40173078542831, # Integral
        9.79714803790873, # Derivative
        dt # Sampling time
    )

    # Save Initial Conditions
    xs.append(x0)
    ys.append(y0)
    vxs.append(v0x)
    vys.append(v0y)

    simulate_wind_field()

    # Discard the last position and velocity 
    # (p[N+1] and v[N+1])
    xs.pop()
    ys.pop()
    vxs.pop()
    vys.pop()

    # Plots
    T = [t*dt for t in range(duration)]
    fig, ax = plt.subplots(1,2)
    ax[0].plot(T,xs,label='Object Position')
    ax[0].plot(T,tr[0,:],'--',label='Reference Position')
    ax[0].title.set_text(r'Position ($x$)')
    ax[0].legend()
    ax[0].set_xlabel(r'$t$ $[s]$')
    ax[0].set_ylabel(r'$x$ $[m]$')
    ax[1].plot(T,ex)
    ax[1].title.set_text(r'Traking error ($e_x$)')
    ax[1].set_xlabel(r'$t$ $[s]$')
    ax[1].set_ylabel(r'$e_x$ $[m]$')
    fig.suptitle(f'{file_name} Trajectory')
    fig.set_size_inches(16,9)
    if save_plots != 'None':
        plt.savefig(f'imgs/{file_name}-trajectory-x-position.png',dpi=300)
    if show_plots != 'None':
        plt.show()

    fig, ax = plt.subplots(1,2)
    ax[0].plot(T,ys,label='Object Position')
    ax[0].plot(T,tr[1,:],'--',label='Reference Position')
    ax[0].title.set_text(r'Position ($y$)')
    ax[0].legend()
    ax[0].set_xlabel(r'$t$ $[s]$')
    ax[0].set_ylabel(r'$y$ $[m]$')
    ax[1].plot(T,ey)
    ax[1].title.set_text(r'Traking error ($e_y$)')
    ax[1].set_xlabel(r'$t$ $[s]$')
    ax[1].set_ylabel(r'$e_y$ $[m]$')
    fig.suptitle(f'{file_name} Trajectory')
    fig.set_size_inches(16,9)
    if save_plots != 'None':
        plt.savefig(f'imgs/{file_name}-trajectoryy-position.png',dpi=300)
    if show_plots != 'None':
        plt.show()

    fig, ax = plt.subplots()
    ax.plot(xs,ys,label='System Trajectory')
    ax.plot(tr[0,:],tr[1,:],'--',label='Trajectory to Track')
    ax.title.set_text(r'Trajectory')
    ax.legend()
    ax.set_xlabel(r'$x$ $[m]$')
    ax.set_ylabel(r'$y$ $[m]$')
    fig.suptitle(f'{file_name} Trajectory')
    fig.set_size_inches(16,9)
    if save_plots != 'None':
        plt.savefig(f'imgs/{file_name}-trajectory-traking.png',dpi=300)
    if show_plots != 'None':
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
    fig.suptitle(f'{file_name} Trajectory')
    fig.set_size_inches(16,9)
    if save_plots != 'None':
        plt.savefig(f'imgs/{file_name}-trajectory-velocity.png',dpi=300)
    if show_plots != 'None':
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
    fig.suptitle(f'{file_name} Trajectory')
    fig.set_size_inches(16,9)
    if save_plots != 'None':
        plt.savefig(f'imgs/{file_name}-trajectory-control-force.png',dpi=300)
    if show_plots != 'None':
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
    fig.suptitle(f'{file_name} Trajectory')
    fig.set_size_inches(16,9)
    if save_plots != 'None':
        plt.savefig(f'imgs/{file_name}-trajectory-wind-force.png',dpi=300)
    if show_plots != 'None':
        plt.show()

    # System Evolution Animation
    if show_plots == 'full':
        fig, ax = plt.subplots()
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        def animation_function(t):
            ax.clear()
            ax.set_xlim([0,width])
            ax.set_ylim([0,height])
            ax.plot(np.NaN, np.NaN, '-', color='none', label='t={0:.2f} s'.format(t*dt))
            ax.plot(tr[0,t],tr[1,t],'o',color='orange',markersize=7,label='Target Distance=[{0:.2f},{0:.2f}] m'.format(ex[t],ey[t])) # Traget Location
            ax.plot(xs[t],ys[t],'bo',markersize=5) # Object Moving
            ax.quiver(xs[t],ys[t],wind_force_x[t],wind_force_y[t],scale=20,width=0.003,color='r',label='Wind Force=[{0:.2f},{0:.2f}] N'.format(wind_force_x[t],wind_force_y[t])) # Wind Force
            ax.quiver(xs[t],ys[t],ctl_forces_x[t],ctl_forces_y[t],scale=20,width=0.003,color="#4DBEEE",label='Control Force=[{0:.2f},{0:.2f}] N'.format(ctl_forces_x[t],ctl_forces_y[t])) # Control Force
            ax.plot(xs[:t],ys[:t],'b')
            ax.plot(tr[0,:t],tr[1,:t],'--',color='orange')
            # Plot fans
            for fan in fans:
                ax.quiver(fan.p0[0],fan.p0[1],fan.u0[0],fan.u0[1],scale=10,color='k')
            ax.legend()

        anim = animation.FuncAnimation(fig,animation_function,frames=duration,interval=1,repeat=False)
        if save_plots == 'full':
            anim.save(f'imgs/{file_name}-trajectory-simulation.gif', fps=30)
        plt.show()

# GP Training

# The model is trained as
# (v(k)-v_h(k),p(k)-p_h(k)) -> f(k) = wind force

points = 50 # Number of used training points 

train_data = torch.FloatTensor(train_data)
train_label_x = torch.FloatTensor(train_label_x)
train_label_y = torch.FloatTensor(train_label_y)
train_label_x_2 = torch.clone(train_label_x)
train_label_y_2 = torch.clone(train_label_y)
# Randomly select a certain number of data points
x_idxs = random.sample(range(0,len(train_data)),points)
y_idxs = random.sample(range(0,len(train_data)),points)
train_data_x = train_data.index_select(0,torch.IntTensor(x_idxs))
train_data_y = train_data.index_select(0,torch.IntTensor(y_idxs))
train_label_x = train_label_x.index_select(0,torch.IntTensor(x_idxs))
train_label_y = train_label_y.index_select(0,torch.IntTensor(y_idxs))

# Build the models
likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
model_x = ExactGPModel(train_data_x, train_label_x, likelihood_x)
model_y = ExactGPModel(train_data_y, train_label_y, likelihood_y)

training_iter = 500
model_x.train()
model_y.train()
likelihood_x.train()
likelihood_y.train()
optimizer_x = torch.optim.Adam(model_x.parameters(), lr=0.01)
optimizer_y = torch.optim.Adam(model_y.parameters(), lr=0.01)
mll_x = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_x, model_x)
mll_y = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_y, model_y)

for i in range(training_iter):
    # Zero gradients from the previous iteration
    optimizer_x.zero_grad()
    optimizer_y.zero_grad()
    # Output from the model
    output_x = model_x(train_data_x)
    output_y = model_y(train_data_y)
    # Compute the loss and backpropagate the gradients
    loss_x = -mll_x(output_x, train_label_x)
    loss_y = -mll_y(output_y, train_label_y)
    loss_x.backward()
    loss_y.backward()
    optimizer_x.step()
    optimizer_y.step()
    # print('Terminated Iteration {} out of {}'.format(i+1,training_iter))

model_x.eval()
model_y.eval()
likelihood_x.eval()
likelihood_y.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    idxs = [i for i in range(0,3000)]
    test_data = train_data.index_select(0,torch.IntTensor(idxs))
    test_labels_x = train_label_x_2.index_select(0,torch.IntTensor(idxs))
    test_labels_y = train_label_y_2.index_select(0,torch.IntTensor(idxs))
    pred_x = model_x(test_data)
    pred_y = model_y(test_data)

# Initialize x plot
fig, ax = plt.subplots()
fig.set_size_inches(16,9)

# Get upper and lower confidence bounds
lower, upper = pred_x.confidence_region()
# Plot training data as black stars
ax.plot(T,test_labels_x,color='orange')
# Plot predictive means as blue line
ax.plot(T,pred_x.mean.numpy(), 'b')
# Shade between the lower and upper confidence bounds
ax.fill_between(T, lower.numpy(), upper.numpy(), alpha=0.5, color='c')
ax.legend(['Real Data','Mean', 'Confidence'])
ax.set_xlabel(r'$t$ $[s]$')
ax.set_ylabel(r'$F_{wx}$')
plt.savefig(f'imgs/wind-gp-test-x.png',dpi=300)

# Initialize y plot
fig, ax = plt.subplots()
fig.set_size_inches(16,9)

# Get upper and lower confidence bounds
lower, upper = pred_y.confidence_region()
# Plot training data as black stars
ax.plot(T,test_labels_y, color='orange')
# Plot predictive means as blue line
ax.plot(T,pred_y.mean.numpy(), 'b')
# Shade between the lower and upper confidence bounds
ax.fill_between(T, lower.numpy(), upper.numpy(), alpha=0.5, color='c')
ax.legend(['Real Data','Mean','Confidence'])
ax.set_xlabel(r'$t$ $[s]$')
ax.set_ylabel(r'$F_{wy}$')
plt.savefig(f'imgs/wind-gp-test-y.png',dpi=300)