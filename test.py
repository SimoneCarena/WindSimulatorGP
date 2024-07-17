import numpy as np
from modules.Quadrotor import Quadrotor
from modules.MPC import MPC

# Initialize the drone dynamics
x0 = np.zeros(10)
dt = 0.01
drone = Quadrotor(dt, x0)

# Define MPC parameters
horizon = 10
Q = np.eye(5)  # Only track the first 5 elements
R = np.eye(4)

# Create the MPC controller
mpc = MPC(drone, horizon, Q, R)

# Define initial state and reference trajectory
x_ref = np.zeros(5)

# Set the reference trajectory
mpc.set_reference(x_ref)

# Solve the MPC problem
u_opt = mpc.solve()

print("Optimal control input:", u_opt)

# Simulate the drone with the optimal control input and wind
wind = np.array([0.1, 0.1, 0.1])  # Example wind disturbance
new_state = drone.simulate(u_opt, wind)
print("New state of the drone:", new_state)
