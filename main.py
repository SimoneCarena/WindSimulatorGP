import matplotlib.pyplot as plt
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


# Argument Parser
# TODO check for argument correctness
parser = argparse.ArgumentParser()
parser.add_argument('--save_plots',default='None')
parser.add_argument('--show_plots',default='None')
parser.add_argument('--test','-t',action='store_true')

# Parse Arguments
args = parser.parse_args()
save_plots = args.save_plots
show_plots = args.show_plots
test = args.test

# Run the simulation for the different trajectories
for file in os.listdir('trajectories'):
    pass

#---------------------------------------------------------------------------#
#                        Gaussian Process Regression                        #
#---------------------------------------------------------------------------#

# The model is trained as
# (v(k)-v_h(k),p(k)-p_h(k)) -> f(k) = wind force

points = 200 # Number of used training points 

train_data_x = torch.FloatTensor(train_data_x)
train_data_y = torch.FloatTensor(train_data_y)
train_label_x = torch.FloatTensor(train_label_x)
train_label_y = torch.FloatTensor(train_label_y)
# Randomly select a certain number of data points
x_idxs = random.sample(range(0,len(train_data_x)),points)
y_idxs = random.sample(range(0,len(train_data_y)),points)
train_data_x = train_data_x.index_select(0,torch.IntTensor(x_idxs))
train_data_y = train_data_y.index_select(0,torch.IntTensor(y_idxs))
train_label_x = train_label_x.index_select(0,torch.IntTensor(x_idxs))
train_label_y = train_label_y.index_select(0,torch.IntTensor(y_idxs))

# Build the models
likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
model_x = ExactGPModel(train_data_x, train_label_x, likelihood_x)
model_y = ExactGPModel(train_data_y, train_label_y, likelihood_y)

training_iter = 100000
model_x.train()
model_y.train()
likelihood_x.train()
likelihood_y.train()
optimizer_x = torch.optim.Adam(model_x.parameters(), lr=0.001)
optimizer_y = torch.optim.Adam(model_y.parameters(), lr=0.001)
mll_x = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_x, model_x)
mll_y = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_y, model_y)

if not test:
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
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Training the Exact GP model on {} random points for {} iterations\n".format(points,training_iter))
        print('[{}{}] {:.2f}% ({}/{} iterations)'.format('='*int(20*(i+1)/training_iter),' '*(20-int(20*(i+1)/training_iter)),100*(i+1)/training_iter,i+1,training_iter))
        print('')
    print('Training Complete!')
else:
    model_x_dict = torch.load('models/model_x.pth')
    model_y_dict = torch.load('models/model_y.pth')
    likelihood_x_dict = torch.load('models/likelihood_x.pth')
    likelihood_y_dict = torch.load('models/likelihood_y.pth')
    model_x.load_state_dict(model_x_dict)
    model_y.load_state_dict(model_y_dict)
    likelihood_x.load_state_dict(likelihood_x_dict)
    likelihood_y.load_state_dict(likelihood_y_dict)

# Prepare test simulation
## Setup the wind field for testing
file = open('configs/wind_field_test.json')
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

for file in os.listdir('test_trajectories'):
    path = 'test_trajectories/'+file
    file_name = Path(file).stem

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

    system = System(m,r,x0,y0,v0x,v0y,dt)
    model = System(m,r,x0,y0,v0x,v0y,dt)
    pid = PID(
            16.255496588371, # Proportional
            6.40173078542831, # Integral
            9.79714803790873, # Derivative
            dt # Sampling time
        )
    # Create arrays to test the GP model
    test_data_x = []
    test_data_y = []
    test_label_x = []
    test_label_y = []

    # Simulate the evolution
    for (t,target) in enumerate(trajectory):
        total_speed = np.array([0,0],dtype=float)
        for fan in fans:
            speed = fan.generate_wind(system.p[0],system.p[1])
            total_speed+=speed
        test_data_x.append([system.p[0],system.p[1]])
        test_data_y.append([system.p[0],system.p[1]])
        # Generate control force
        error = target - system.p
        control_force = pid.step(error)
        # Generate wind force
        wind_force = (0.5*air_density*system.surf)*total_speed**2*np.sign(total_speed)
        # Total force
        force = wind_force + control_force
        # Simulate Dynamics
        system.discrete_dynamics(force)

        test_label_x.append(wind_force[0])
        test_label_y.append(wind_force[1])


    test_data_x = torch.FloatTensor(test_data_x)
    test_data_y = torch.FloatTensor(test_data_y)
    test_label_x = torch.FloatTensor(test_label_x)
    test_label_y = torch.FloatTensor(test_label_y)

    # Test the model
    model_x.eval()
    model_y.eval()
    likelihood_x.eval()
    likelihood_y.eval()

    with torch.no_grad():
        # pred_x = likelihood_x(model_x(test_data_x))
        # pred_y = likelihood_y(model_y(test_data_y))
        pred_x = model_x(test_data_x)
        pred_y = model_y(test_data_y)

    kernel = 'RBF-Periodic'

    # Initialize x plot
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    rmse = np.sqrt(1/len(pred_x.mean)*np.linalg.norm(pred_x.mean.numpy()-test_label_x.numpy())**2)

    # Get upper and lower confidence bounds
    lower, upper = pred_x.confidence_region()
    ax.plot(np.NaN, np.NaN, '-', color='none', label='RMSE={0:.2f} N'.format(rmse))
    # Plot training data as black stars
    ax.plot(T,test_label_x,color='orange',label='Real Data')
    # Plot predictive means as blue line
    ax.plot(T,pred_x.mean.numpy(), 'b',label='Mean')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(T, lower.numpy(), upper.numpy(), alpha=0.5, color='c',label='Confidence')
    # Plot training points
    ax.plot([T[idx] for idx in x_idxs],train_label_x,'k*',label='Training Points')
    ax.legend()
    ax.set_xlabel(r'$t$ $[s]$')
    ax.set_ylabel(r'$F_{wx}$ $[N]$')
    fig.suptitle(f'GP Wind Estimation on {file_name} Trajectory (x-axis)')
    if save_plots != 'None':
        plt.savefig(f'imgs/test_imgs/{file_name}-{kernel}-x.png',dpi=300)
    plt.show()

    # Initialize y plot
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    rmse = np.sqrt(1/len(pred_y.mean)*np.linalg.norm(pred_y.mean.numpy()-test_label_y.numpy())**2)

    # Get upper and lower confidence bounds
    lower, upper = pred_y.confidence_region()
    ax.plot(np.NaN, np.NaN, '-', color='none', label='RMSE={0:.2f} N'.format(rmse))
    # Plot training data as black stars
    ax.plot(T,test_label_y, color='orange',label='Real Data')
    # Plot predictive means as blue line
    ax.plot(T,pred_y.mean.numpy(), 'b',label='Mean')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(T, lower.numpy(), upper.numpy(), alpha=0.5, color='c',label='Confidence')
    # Plot training points
    ax.plot([T[idx] for idx in y_idxs],train_label_y,'k*',label='Training Points')
    ax.legend()
    ax.set_xlabel(r'$t$ $[s]$')
    ax.set_ylabel(r'$F_{wy}$ $[N]$')
    fig.suptitle(f'GP Wind Estimation on {file_name} Trajectory (y-axis)')
    if save_plots != 'None':
        plt.savefig(f'imgs/test_imgs/{file_name}-{kernel}-y.png',dpi=300)
    plt.show()
    plt.close()

if not test:
    # Save the models
    torch.save(model_x.state_dict(), 'models/model_x.pth')
    torch.save(model_y.state_dict(), 'models/model_y.pth')
    torch.save(likelihood_x.state_dict(), 'models/likelihood_x.pth')
    torch.save(likelihood_y.state_dict(), 'models/likelihood_y.pth')