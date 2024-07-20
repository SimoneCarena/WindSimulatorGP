import numpy as np
import casadi as ca

class Trajectory:
    def __init__(self, dynamics, dt, N):
        self.dynamics = dynamics
        self.dt = dt
        self.N = N
        self.nx = 10
        self.nu = 4
        self.__setup_solver()

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

    def __setup_solver(self):
        # Setup Optimization Problem
        opti = ca.Opti()

        # Define optimization variables
        u = opti.variable(self.nu, self.N) # Control Variable
        x = opti.variable(self.nx, self.N + 1) # State Variable
        x0 = opti.parameter(self.nx ) # Initial State
        ref = opti.parameter(3, self.N)
        t = opti.variable() # Total Time

        self.u = u
        self.x = x
        self.x0 = x0
        self.t = t
        self.ref = ref

        # Setup Minimization Probelm
        opti.minimize(t)
        dt = t/self.N

        for k in range(self.N):
            # Current Reference
            x_ref = ref[:,k] 
            # Compute Dynamics
            x_next = self.__step(x[:,k],u[:,k],dt) 
            # Add Dynamics Constraint   
            opti.subject_to(x[:,k+1]==x_next)
            # Control Constraints (Upper Bound)
            opti.subject_to(
                u[:,k]<np.array([np.pi/6,np.pi/6,np.pi/6,15])
            )
            # Control Constraints (Lower Bound)
            opti.subject_to(
                u[:,k]>np.array([-np.pi/6,-np.pi/6,-np.pi/6,5])
            )
        
        # Add Initial State Constraint
        opti.subject_to(x[:,0]==x0)
        # Add Final Position Constraint
        opti.subject_to(x[:3,self.N]==ref[:,-1])
        # Add Time Constraint
        opti.subject_to(t>0)

        opti.solver('ipopt')
        self.solver = opti

    def __call__(self, x0, ref):
        # Initial guess
        u_guess = np.zeros((self.nu, self.N))
        x_guess = np.zeros((self.nx, self.N + 1))

        # Set Initial Guess 
        self.solver.set_initial(self.t,1)
        self.solver.set_initial(self.x,x_guess)
        self.solver.set_initial(self.u,u_guess)

        # Set Initial Condition and Reference
        self.solver.set_value(self.x0,x0)
        self.solver.set_value(self.ref,ref)

        # Solve the Problem
        sol = self.solver.solve()

        # Extract the solution
        x_opt = sol.value(self.x)
        u_opt = sol.value(self.u)
        t_opt = sol.value(self.t)
        
        return np.array(x_opt), np.array(u_opt), np.array(t_opt)    