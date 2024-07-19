import numpy as np
import casadi as ca

class MPC:
    def __init__(self, dynamics, control_horizon, dt, Q, R):
        self.dynamics = dynamics
        self.N = control_horizon
        self.dt = dt

        self.nx = 10 # State Dimension
        self.ny = 4 # Tracking Dimension
        self.nu = 4 # Control Dimension

        # Define the cost function to focus on the first 5 state elements
        self.Q = Q  
        self.R = R
        self.__setup_solver()

    def __setup_solver(self):
        # Setup Optimization Problem
        opti = ca.Opti()

        # Define optimization variables
        self.u = opti.variable(self.nu, self.N) # Control Variable
        self.x = opti.variable(self.nx, self.N + 1) # State Variable
        self.x0 = opti.parameter(self.nx) # Initial State
        self.ref = opti.parameter(self.ny, self.N)

        # Define cost variable
        self.J = 0.0

        for k in range(self.N):
            # Compute Cost
            self.J += (self.x[6:,k]-self.ref[:,k]).T@self.Q@(self.x[6:,k]-self.ref[:,k])
            # Compute Dynamics
            x_next = self.__step(self.x[:,k],self.u[:,k],self.dt) 
            # Add Dynamics Constraint   
            opti.subject_to(self.x[:,k+1]==x_next)
            # Control Constraints (Upper Bound)
            opti.subject_to(
                self.u[:,k]<np.array([np.pi/6,np.pi/6,np.pi/6,15])
            )
            # Control Constraints (Lower Bound)
            opti.subject_to(
                self.u[:,k]>np.array([-np.pi/6,-np.pi/6,-np.pi/6,5])
            )
        
        # Add Initial State Constraint
        opti.subject_to(self.x[:,0] == self.x0)

        # Setup Optimization Problem
        opti.minimize(self.J)

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        opti.solver('ipopt',opts)
        self.solver = opti

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
        # Initial guess
        u_guess = np.zeros((self.nu, self.N))
        x_guess = np.zeros((self.nx, self.N + 1))

        # Set Initial Guess 
        self.solver.set_initial(self.x,x_guess)
        self.solver.set_initial(self.u,u_guess)

        # Set Initial Condition and Reference
        self.solver.set_value(self.x0,x)
        self.solver.set_value(self.ref,ref)

        # Solve the Problem
        sol = self.solver.solve()

        # Extract the solution
        u_opt = sol.value(self.u)
        
        return np.array(u_opt[:,0]).flatten()