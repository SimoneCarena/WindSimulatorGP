import numpy as np
import casadi as ca

class MPC:
    def __init__(self, dynamics, control_horizon, dt, Q, R, input_dim ,output_dim, window_size, predictor=None):
        self.dynamics = dynamics
        self.N = control_horizon
        self.dt = dt
        self.predictor = predictor
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.lower = np.array([-np.pi/6, -np.pi/6, -np.pi/6, 5])
        self.upper = np.array([np.pi/6, np.pi/6, np.pi/6, 15])

        self.nx = 10  # State Dimension
        self.ny = 6  # Tracking Dimension
        self.nu = 4  # Control Dimension

        # Define the cost function to focus on the first 5 state elements
        self.Q = Q
        self.R = R
        self.__setup_solver()

    def __setup_solver(self):
        # Setup Optimization Problem
        self.opti = ca.Opti()

        # Define optimization variables
        self.u = self.opti.variable(self.nu, self.N)  # Control Variable
        self.x = self.opti.variable(self.nx, self.N + 1)  # State Variable
        self.x0 = self.opti.parameter(self.nx)  # Initial State
        self.ref = self.opti.parameter(self.ny, self.N)
        # GP parameters
        self.X = self.opti.parameter(self.input_dim,self.window_size) # Inputs
        self.y = self.opti.parameter(self.window_size,self.output_dim) # Labels
        self.K = self.opti.parameter(self.window_size,self.window_size) # Inverse of the kernel matrix

        # Define cost variable
        self.J = 0.0

        # Add Initial State Constraint
        self.opti.subject_to(self.x[:, 0] == self.x0)

        for k in range(self.N):
            # Compute Cost
            self.J += (self.x[:self.ny, k] - self.ref[:, k]).T @ self.Q @ (self.x[:self.ny, k] - self.ref[:, k]) + self.u[:,k].T@self.R@self.u[:,k]
            # Compute Dynamics
            ## If there is a predictor compute the wind prediction
            if self.predictor is not None:
                # Compute GP Predictive Distribution
                ## Compute the correlation between the new input and the past values
                self.K_xx = []
                for j in range(self.window_size):
                    self.K_xx = ca.vertcat(self.K_xx,self.predictor.kernel(self.x[:self.input_dim,k],self.X[:,j]))
                mean = self.K_xx.T@self.K@self.y
                # cov = self.predictor.kernel(self.ref[:2,k],self.ref[:2,k])-self.K_xx.T@self.K@self.K_xx
                x_next = self.__step(self.x[:, k], self.u[:, k], mean, self.dt)
            ## Otherwise compute the dynamics without the wind
            else:
                x_next = self.__step(self.x[:, k], self.u[:, k], ca.MX.zeros(3), self.dt) 
            # Add Dynamics Constraint
            self.opti.subject_to(self.x[:, k + 1] == x_next)
            # Control Constraints (Upper Bound)
            self.opti.subject_to(self.u[:, k] <= self.upper)
            # Control Constraints (Lower Bound)
            self.opti.subject_to(self.u[:, k] >= self.lower)

        # Setup Optimization Problem
        self.opti.minimize(self.J)

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver(
            'ipopt', 
            opts
        )

    def __step(self, x, u, wind, dt):
        """
        Perform a single RK4 integration step.
        :param x: Current state
        :param u: Control input
        :param dt: Time step
        :return: State after one RK4 step
        """
        k1 = self.dynamics(x, u, wind)
        k2 = self.dynamics(x + dt / 2 * k1, u, wind)
        k3 = self.dynamics(x + dt / 2 * k2, u, wind)
        k4 = self.dynamics(x + dt * k3, u, wind)
        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        return x_next

    def __call__(self, x, ref):
        # Set Initial Guess
        # Initial guess
        u_guess = np.repeat(
            (self.lower[:, np.newaxis] + self.upper[:, np.newaxis]) / 2,
            self.N,
            axis=1
        )
        x_guess = np.repeat(
            x[:,np.newaxis],
            self.N+1,
            axis=1
        )
        self.opti.set_initial(self.x, x_guess)
        self.opti.set_initial(self.u, u_guess)

        # Set Initial Condition and Reference
        self.opti.set_value(self.x0, x)
        self.opti.set_value(self.ref, ref)

        # Set GP Data
        if self.predictor is not None:
            K, inputs, labels = self.predictor()
        else:
            K = np.zeros((self.window_size,self.window_size))
            inputs = np.zeros((self.input_dim,self.window_size))
            labels = np.zeros((self.window_size,self.output_dim))
        self.opti.set_value(self.K, K)
        self.opti.set_value(self.X, inputs)
        self.opti.set_value(self.y, labels)

        # Solve the Problem
        sol = self.opti.solve()

        # Extract the solution
        u_opt = sol.value(self.u)
        x_opt = sol.value(self.x)
        
        return np.array(u_opt[:, 0]).flatten(), np.array(x_opt[:, 1:])

    def update_predictor(self, input, label):
        self.predictor.update(
            input,
            label
        )