import numpy as np
import casadi as ca

class MPCController:
    def __init__(self, quadrotor, prediction_horizon, control_horizon, dt, Q, R):
        self.quadrotor = quadrotor
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.dt = dt

        # Define optimization variables
        self.u = ca.SX.sym('u', 4, control_horizon)
        self.p = ca.SX.sym('p',10+4) # Initial State + Reference

        self.nx = 10 # State Dimension
        self.ny = 4 # Tracking Dimension
        self.nu = 4 # Control Dimension

        # Define the cost function to focus on the first 5 state elements
        self.Q = Q  
        self.R = R
        self.define_optimization_problem()

    def define_optimization_problem(self):
        cost = 0
        constraints = []

        # Initialize the state trajectory
        x_t = self.p[:10]
        ref = self.p[10:]
        for t in range(self.control_horizon):
            # Compute cost
            cost += (ref-x_t[6:]).T@self.Q@(ref-x_t[6:]) + self.u[:,t].T@self.R@self.u[:,t]

            # Compute next state using RK4 integration
            x_next = self.rk4_integration(x_t, self.u[:, t], self.dt)
            x_t = x_next

        # Formulate the NLP
        nlp = {'x': ca.reshape(self.u, -1, 1), 'f': cost, 'g': ca.vertcat(*constraints), 'p': self.p}
        opts = {'ipopt.print_level':0, 'print_time':0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    def rk4_integration(self, x, u, dt):
        """
        Perform a single RK4 integration step.
        
        :param x: Current state
        :param u: Control input
        :param dt: Time step
        :return: State after one RK4 step
        """
        k1 = self.quadrotor.ideal_dynamics(x, u)
        k2 = self.quadrotor.ideal_dynamics(x + dt/2 * k1, u)
        k3 = self.quadrotor.ideal_dynamics(x + dt/2 * k2, u)
        k4 = self.quadrotor.ideal_dynamics(x + dt * k3, u)
        x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return x_next

    def compute_control_input(self, x0, ref):
        # Solve the optimization problem
        initial_guess = np.zeros(4 * self.control_horizon)
        sol = self.solver(x0=initial_guess, p=ca.vertcat(x0,ref), lbg=0, ubg=0)
        u_opt = sol['x'].reshape((4, self.control_horizon))
        return u_opt[:, 0]