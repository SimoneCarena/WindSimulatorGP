import numpy as np
from modules.Quadrotor import Quadrotor
from modules.MPC import MPCController

# Instantiate the Quadrotor and MPCController
dt = 0.1
x0 = np.zeros(10)
quadrotor = Quadrotor(dt, x0)
mpc_controller = MPCController(quadrotor, prediction_horizon=10, control_horizon=5, dt=dt, Q=np.eye(4), R=np.eye(4))

# Example usage in a simulation loop
time_steps = 100
control_inputs = []
states = [x0]

for t in range(time_steps):
    x_current = states[-1]
    u = mpc_controller.compute_control_input(x_current,np.array([0,10,0,11]))
    print(u)
    control_inputs.append(u)
    quadrotor.step(u, np.zeros(3))  # Assuming no wind disturbance for ideal dynamics
    states.append(quadrotor.get_state())

control_inputs, states