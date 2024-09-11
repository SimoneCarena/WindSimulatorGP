import casadi as ca
import numpy as np

class Quadrotor:
    def __init__(self, dt, x0):
        tau = np.array([0.18,0.18,0.56,0.05])
        k = 1
        self.dt = dt
        self.state = x0
        self.Tau = np.diag(-1/tau)
        self.K = k/tau
        self.g = np.array([0.0,0.0,9.81])
        self.nx = 10 # Number of States
        self.nu = 4 # Number of Control Inputs
        self.r = 0.1 # Radius
        self.__setup_dynamics()

    def __dynamics(self,x,u,wind):
        p = x[:3]
        v = x[3:6]
        att = x[6:]

        # Define the dynamics
        p_dot = v

        phi = att[0]
        theta = att[1]
        psi = att[2]
        thrust = att[3]

        v_dot = np.array([
            np.sin(phi)*np.sin(psi) + np.cos(phi)*np.sin(theta)*np.cos(psi),
            -np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi),
            np.cos(phi)*np.cos(psi)
        ]) * thrust - self.g + wind

        att_dot = self.Tau @ att + self.K * u

        # Concatenate the state derivatives
        state_dot = np.concatenate([p_dot.flatten(), v_dot.flatten(), att_dot.flatten()])

        return state_dot

    def get_dynamics(self):
       return self.dynamics_function
    
    def get_diff_dynamics(self):
        return self.diff_dynamics_function
    
    def __setup_dynamics(self):
         # Define the state variables
        p = ca.SX.sym('p', 3)       # Position [x, y, z]
        v = ca.SX.sym('v', 3)       # Velocity [vx, vy, vz]
        att = ca.SX.sym('att', 4)   # Attitude [phi, theta, psi, thrust]
        state = ca.vertcat(p, v, att)

        # Control Inputs
        u = ca.SX.sym('u', 4) # Control input [phi_c, theta_c, psi_c, thrust_c]
        # Wind Force
        wind = ca.SX.sym('wind',3)

        # Define the dynamics
        p_dot = v

        phi = att[0]
        theta = att[1]
        psi = att[2]
        thrust = att[3]

        v_dot = ca.vertcat(
            ca.sin(phi)*ca.sin(psi) + ca.cos(phi)*ca.sin(theta)*ca.cos(psi),
            -ca.sin(phi)*ca.cos(psi) + ca.cos(phi)*ca.sin(theta)*ca.sin(psi),
            ca.cos(phi)*ca.cos(psi)
        ) * thrust - self.g + wind

        att_dot = self.Tau @ att + self.K * u

        # Concatenate the state derivatives
        state_dot = ca.vertcat(p_dot, v_dot, att_dot)

        # Define CasADi function
        self.dynamics_function = ca.Function('dynamics', [state, u, wind], [state_dot])
        
        jacobian = ca.jacobian(state_dot,state)
        self.diff_dynamics_function = ca.Function('diff_dynamics', [state, u, wind], [jacobian])

    
    def step(self, u, wind):
        """
        Perform a single RK4 integration step.
        
        :param x: Current state
        :param u: Control input
        :param dt: Time step
        :return: State after one RK4 step
        """
        x = self.state

        k1 = self.__dynamics(x, u, wind)
        k2 = self.__dynamics(x + self.dt/2 * k1, u, wind)
        k3 = self.__dynamics(x + self.dt/2 * k2, u, wind)
        k4 = self.__dynamics(x + self.dt * k3, u, wind)
        x_next = x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        self.state = x_next

    def set_state(self, new_state):
        self.state = new_state

    def get_state(self) -> np.array:
        return self.state

    def get_dimensions(self) -> tuple[int,int]:
        return self.nx, self.nu