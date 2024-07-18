import numpy as np
import casadi as ca

class MPC:
    def __init__(self, dynamics, control_horizon, dt, Q, R):
        self.dynamics = dynamics
        self.control_horizon = control_horizon
        self.dt = dt

        # Define optimization variables
        self.u = ca.SX.sym('u', 4, control_horizon)
        self.p = ca.SX.sym('p',10+6) # Initial State + Reference

        self.nx = 10 # State Dimension
        self.ny = 6 # Tracking Dimension
        self.nu = 4 # Control Dimension

        # Define the cost function to focus on the first 5 state elements
        self.Q = Q  
        self.R = R
        self.define_optimization_problem()

    def define_optimization_problem(self):
        cost = 0
        # Equality Constratints
        g = []
        # Inequality Constraints on the optimization variable
        lower_bound = []
        upper_bound = []

        # Initialize the state trajectory
        x_t = self.p[:10]
        ref = self.p[10:]
        for t in range(self.control_horizon):
            # Compute cost
            cost += (ref-x_t[:self.ny]).T@self.Q@(ref-x_t[:self.ny]) + self.u[:,t].T@self.R@self.u[:,t]

            # Compute next state using RK4 integration
            x_next = self.__step(x_t, self.u[:, t], self.dt)
            x_t = x_next

            lower_bound += [-np.pi/6,-np.pi/6,-np.pi/6,5]
            upper_bound += [np.pi/6,np.pi/6,np.pi/6,15]

        self.lower_bound = ca.vertcat(lower_bound)
        self.upper_bound = ca.vertcat(upper_bound)

        # Formulate the NLP
        nlp = {'x': ca.reshape(self.u, -1, 1), 'f': cost, 'g': ca.vertcat(*g), 'p': self.p}
        opts = {'ipopt.print_level':0, 'print_time':0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp)

    def __step(self, x, u, dt):
        """
        Perform a single RK4 integration step.
        
        :param x: Current state
        :param u: Control input
        :param dt: Time step
        :return: State after one RK4 step
        """
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + dt/2 * k1, u)
        k3 = self.dynamics(x + dt/2 * k2, u)
        k4 = self.dynamics(x + dt * k3, u)
        x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return x_next

    def __call__(self, x, ref):
        # Solve the optimization problem
        initial_guess = np.zeros(self.nu * self.control_horizon)
        sol = self.solver(
            x0=initial_guess, 
            p=ca.vertcat(x,ref), 
            lbx=self.lower_bound, 
            ubx=self.upper_bound,
            lbg = 0,
            ubg = 0
        )
        u_opt = sol['x'].reshape((self.nu, self.control_horizon))
        return u_opt[:, 0]