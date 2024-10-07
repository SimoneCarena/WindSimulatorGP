import numpy as np
import casadi as ca
import sys

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import vertcat
from scipy.stats import chi2
from scipy.special import erfinv

class MPC:
    def __init__(self, model, control_horizon, dt, Q, R, input_dim, output_dim, window_size,obstacles=[],predictor=None):
        self.model = model
        self.predictor = predictor
        self.N = control_horizon
        self.dt = dt
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.obstacles = obstacles
        self.quadrotor_r = 0.1
        self.delta = 0.05

        self.lower = np.array([-np.pi/6, -np.pi/6, -np.pi/6, 5])
        self.upper = np.array([np.pi/6, np.pi/6, np.pi/6, 15])

        self.Q = Q
        self.R = R

        self.ny = 6  # Dimension of the controlled state
        self.nx = 10 # Dimensions of the state
        self.nu = 4  # Dimension of the inputs

        if self.predictor is not None:
            self.__setup_solver_gp()
        else:
            self.__setup_solver()

    def __setup_solver(self):
        ocp = AcadosOcp()

        # Use the previously created acados model
        model = AcadosModel()
        state, u, state_dot = self.model.get_acados_model()
        model.x = state
        model.u = u
        model.f_expl_expr = state_dot
        model.name = 'quadrotor_mpc'
        ocp.model = model

        # Time horizon and discretization
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt

        # Cost function
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ## Weight matrix for intermediate steps
        ocp.cost.W = ca.diagcat(self.Q,self.R).full()
        ## Weight matrix for final step
        ocp.cost.W_e = self.Q
        ## Mapping matrices:
        ### - x -> state
        ### - u -> inputs
        ### _e -> final state/input, otherwise, intermediate
        ocp.cost.Vx = np.diag([1,1,1,1,1,1,0,0,0,0])
        ocp.cost.Vx_e = np.block([
            np.eye(self.ny), np.zeros((self.ny,self.nx-self.ny))
        ])
        ocp.cost.Vu = np.block([
            [np.zeros((self.ny,self.nu))],
            [np.eye(self.nu)]
        ])
        ## The reference is shaped (ny+nu), and is the concatenations of the
        ## state reference (ny as the nyumber of controlled state != the number of total states)
        ## and the input reference (nu)
        ocp.cost.yref = np.array([2,2,2,0,0,0,0,0,0,0])
        ## The terminal cost only referencec the state
        ocp.cost.yref_e = np.array([2,2,2,0,0,0])

        # Constraints
        ocp.constraints.lbu = self.lower
        ocp.constraints.ubu = self.upper
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.x0 = np.zeros(self.nx)

        # Add Obstacles avoidance constraints
        ## The constraints on the obstacles must be added only if there are obstacles 
        ## otherwise casadi complains if we add an empy array
        if self.obstacles:
            num_obstacles = len(self.obstacles)
            h = []
            for obstacle in self.obstacles:
                p0 = obstacle.p
                r = obstacle.r
                h = vertcat(
                    h,
                    (ocp.model.x[:2]-p0).T@(ocp.model.x[:2]-p0)-(r+self.quadrotor_r)**2
                )
            ## The formulation for the constraints has to be expressed as
            ## lh_i <= h_i(x,u) <= uh_i
            ## for each obstacle i
            ## Some value must be set as an upper bound for the optimization, even though
            ## only the formulation h(x,u) <= 0 is needed, thus an upper bound of 
            ## 1000 is used (which is reasonably large)
            ocp.model.con_h_expr_0 = h
            ocp.model.con_h_expr = h
            ocp.constraints.lh_0 = np.zeros(num_obstacles)
            ocp.constraints.uh_0 = np.array([1000]*num_obstacles)
            ocp.constraints.lh = np.zeros(num_obstacles)
            ocp.constraints.uh = np.array([1000]*num_obstacles)

        # Create the solver
        ocp.code_export_directory = 'acados/solver_no_gp'
        self.solver = AcadosOcpSolver(ocp, json_file='acados/acados_ocp.json')

    def __setup_solver_gp(self):
        ocp = AcadosOcp()

        # Use the previously created acados model
        model = AcadosModel()
        mean = ca.SX.sym('mean',3)
        state, u, state_dot = self.model.get_acados_model()
        model.x = state
        model.u = u
        state_dot[3:6]+=mean
        model.f_expl_expr = state_dot
        model.p = mean
        model.name = 'quadrotor_gp_mpc'

        ocp.model = model
        ocp.parameter_values = np.zeros((
            3,
            1
        ))

        # Time horizon and discretization
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt

        # Cost function
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ## Weight matrix for intermediate steps
        ocp.cost.W = ca.diagcat(self.Q,self.R).full()
        ## Weight matrix for final step
        ocp.cost.W_e = self.Q
        ## Mapping matrices:
        ### - x -> state
        ### - u -> inputs
        ### _e -> final state/input, otherwise, intermediate
        ocp.cost.Vx = np.diag([1,1,1,1,1,1,0,0,0,0])
        ocp.cost.Vx_e = np.block([
            np.eye(self.ny), np.zeros((self.ny,self.nx-self.ny))
        ])
        ocp.cost.Vu = np.block([
            [np.zeros((self.ny,self.nu))],
            [np.eye(self.nu)]
        ])
        ## The reference is shaped (ny+nu), and is the concatenations of the
        ## state reference (ny as the nyumber of controlled state != the number of total states)
        ## and the input reference (nu)
        ocp.cost.yref = np.array([0,0,0,0,0,0,0,0,0,0])
        ## The terminal cost only referencec the state
        ocp.cost.yref_e = np.array([0,0,0,0,0,0])

        # Constraints
        ocp.constraints.lbu = self.lower
        ocp.constraints.ubu = self.upper
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.x0 = np.zeros(self.nx)

        # Add Obstacles avoidance constraints
        ## The constraints on the obstacles must be added only if there are obstacles 
        ## otherwise casadi complains if we add an empy array
        if self.obstacles:
            num_obstacles = len(self.obstacles)
            h = []
            for obstacle in self.obstacles:
                p0 = obstacle.p
                r = obstacle.r
                h = vertcat(
                    h,
                    (ocp.model.x[:2]-p0).T@(ocp.model.x[:2]-p0)-(r+self.quadrotor_r)**2
                )
            ## The formulation for the constraints has to be expressed as
            ## lh_i <= h_i(x,u) <= uh_i
            ## for each obstacle i
            ## Some value must be set as an upper bound for the optimization, even though
            ## only the formulation h(x,u) <= 0 is needed, thus an upper bound of 
            ## 1000 is used (which is reasonably large)
            ocp.model.con_h_expr_0 = h
            ocp.model.con_h_expr = h
            ocp.constraints.lh_0 = np.zeros(num_obstacles)
            ocp.constraints.uh_0 = np.array([1000]*num_obstacles)
            ocp.constraints.lh = np.zeros(num_obstacles)
            ocp.constraints.uh = np.array([1000]*num_obstacles)

        # Create the solver
        ocp.code_export_directory = 'acados/solver_gp'
        self.solver = AcadosOcpSolver(ocp, json_file='acados/acados_ocp.json')

    def __call__(self, x, ref, prev_x_opt):
        """
        Solve the MPC optimization problem.

        :param x: The current state (initial condition), shape (nx,).
        :param ref: The reference trajectory, shape (nx, N+1).
        :return: Optimal control input at the first step, predicted state trajectory.
        """
        # Set the initial guess for state and control
        u_guess = np.tile((self.lower + self.upper) / 2, (self.N, 1))  # Initial guess for controls
        x_guess = np.tile(x, (self.N + 1, 1))  # Initial guess for states

        # Set initial guess in the solver
        for k in range(self.N):
            self.solver.set(k, "x", x_guess[k, :])  # Guess for states
            self.solver.set(k, "u", u_guess[k, :])  # Guess for controls

        self.solver.set(self.N, "x", x_guess[self.N, :])  # Guess for last state
        # Set initial condition constraint
        self.solver.set(0, "lbx", x)
        self.solver.set(0, "ubx", x)

        # Set the reference trajectory for each time step
        # Each reference in `ref[:, k]` corresponds to time step k
        for k in range(self.N):
            self.solver.set(
                k, 
                "yref",
                np.concatenate((ref[:,k], np.array([0,0,0,0])))
            )

        # Set final refrence
        self.solver.set(
            self.N,
            "yref",
            ref[:,-1]
        )

        if self.predictor is not None:
            K_inv, X, y = self.predictor()
            for k in range(self.N):
                K_xx = []
                for t in range(self.window_size):
                    K_xx.append(
                        self.predictor.kernel(
                            prev_x_opt[:2,k],
                            X[:,t]
                        )
                    )
                K_xx = np.array(ca.vertcat(*K_xx))
                mean = K_xx.T@K_inv@y
                self.solver.set(
                    k,
                    "p",
                    mean.T
                )
            # print('')

        # Solve the optimization problem
        status = self.solver.solve()

        # Check the solver status
        if status != 0:
            print(
                '\033[93m'+f"Solver Returned Exit Status {4}"+'\033[93m',
                file = sys.stderr
            )

        # Extract the solution: control inputs and predicted states
        u_opt = np.zeros((self.N, self.nu))
        x_opt = np.zeros((self.N + 1, self.nx))
        for k in range(self.N):
            u_opt[k, :] = self.solver.get(k, "u")  # Optimal control at each step
            x_opt[k, :] = self.solver.get(k, "x")  # Predicted state at each step

        x_opt[self.N, :] = self.solver.get(self.N, "x")  # Last state in prediction horizon

        # Return the first control input and the predicted state trajectory
        if self.predictor is None:
            return u_opt[0, :], x_opt.T, None
        else:
            return u_opt[0, :], x_opt.T, np.zeros((2*self.window_size,2))
    
    def update_predictor(self, input, label):
        self.predictor.update(
            input,
            label
        )
  

# class MPC:
#     def __init__(self, model, control_horizon, dt, Q, R, input_dim ,output_dim, window_size, predictor=None, obstacles = []):
#         self.dynamics = model.get_dynamics()
#         self.diff_dynamics = model.get_diff_dynamics()
#         self.N = control_horizon
#         self.dt = dt
#         self.predictor = predictor
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.window_size = window_size
#         self.lower = np.array([-np.pi/6, -np.pi/6, -np.pi/6, 5])
#         self.upper = np.array([np.pi/6, np.pi/6, np.pi/6, 15])
#         # Eigenvectors for the direction of the ellipse,
#         # assumed to be identical over all the directions
#         self.eigs = np.eye(output_dim)
#         self.obstacles = obstacles
#         self.quadrotor_r = 0.1
#         self.delta = 0.05
#         self.unc_val = erfinv(1-2*self.delta) 
#         self.chi2_val = chi2.ppf(1-self.delta, 2)

#         self.nx = 10  # State Dimension
#         self.ny = 6  # Tracking Dimension
#         self.nu = 4  # Control Dimension

#         # Define the cost function to focus on the first 5 state elements
#         self.Q = Q
#         self.R = R
#         self.__setup_solver()

#     def __setup_solver(self):
#         # Setup Optimization Problem
#         self.opti = ca.Opti()

#         # Define optimization variables
#         self.u = self.opti.variable(self.nu, self.N)  # Control Variable
#         self.x = self.opti.variable(self.nx, self.N + 1)  # State Variable
#         self.x0 = self.opti.parameter(self.nx)  # Initial State
#         self.ref = self.opti.parameter(self.ny, self.N)
#         # GP parameters
#         self.X = self.opti.parameter(self.input_dim,self.window_size) # Inputs
#         self.y = self.opti.parameter(self.window_size,self.output_dim) # Labels
#         self.K = self.opti.parameter(self.window_size,self.window_size) # Inverse of the kernel matrix

#         # Define cost variable
#         self.J = 0.0

#         # Add Initial State Constraint
#         self.opti.subject_to(self.x[:, 0] == self.x0)

#         cov_x = np.zeros((2,2))
#         Cov = []

#         for k in range(self.N):
#             # Compute Cost
#             self.J += (self.x[:self.ny, k] - self.ref[:, k]).T @ self.Q @ (self.x[:self.ny, k] - self.ref[:, k]) + self.u[:,k].T@self.R@self.u[:,k]
#             # Compute Dynamics
#             ## If there is a predictor compute the wind prediction
#             if self.predictor is not None:
#                 # Compute GP Predictive Distribution
#                 ## Compute the correlation between the new input and the past values
#                 self.K_xx = []
#                 for j in range(self.window_size):
#                     self.K_xx = ca.vertcat(self.K_xx,self.predictor.kernel(self.x[:self.input_dim,k],self.X[:,j]))
#                 mean = self.K_xx.T@self.K@self.y
#                 cov_gp = self.predictor.kernel(self.x[:2,k],self.x[:2,k])-self.K_xx.T@self.K@self.K_xx
#                 cov_gp = cov_gp*np.eye(2)
#                 x_next = self.__step(self.x[:, k], self.u[:, k], mean, self.dt)
#                 # Propagate Uncertainty
#                 A = self.diff_dynamics(self.x[:,k],self.u[:,k],mean)
#                 A = A[:2,:2]
#                 K_xx_d = []
#                 for j in range(self.window_size):
#                     K_xx_d = ca.vertcat(K_xx_d,self.predictor.kernel_derivative(self.x[:self.input_dim,k],self.X[:,j]).T)
#                 mean_d = K_xx_d.T@self.K@self.y[:,:2]
#                 Sigma_xd = mean_d@cov_x
#                 _first_mat = ca.horzcat(A,np.eye(2))
#                 _second_mat = ca.vertcat(
#                     ca.horzcat(cov_x, Sigma_xd),
#                     ca.horzcat(Sigma_xd.T, cov_gp)
#                 )
#                 _third_mat = _first_mat.T
#                 cov = _first_mat@ _second_mat@ _third_mat
#                 # Compute the semi-axis of the ellipse
#                 lx = ca.sqrt(cov[0,0]*self.chi2_val)
#                 ly = ca.sqrt(cov[1,1]*self.chi2_val)
#                 # Approximate the ellipse with a circle of radius equal to
#                 # the maximum of the 2 semi-axis
#                 l = ca.fmax(lx,ly)
#                 # Compute ellipsoidal constraints
#                 for obstacle in self.obstacles:
#                     r = obstacle.r
#                     p0 = obstacle.p
#                     d2 = (r+self.quadrotor_r+l)**2
#                     self.opti.subject_to(
#                         (self.x[:2,k+1]-p0).T@(self.x[:2,k+1]-p0)>d2
#                     )
#                 cov_x = cov
#                 Cov = ca.vertcat(Cov,cov)
#             ## Otherwise compute the dynamics without the wind
#             else:
#                 x_next = self.__step(self.x[:, k], self.u[:, k], ca.MX.zeros(3), self.dt) 
#                 for obstacle in self.obstacles:
#                     r = obstacle.r
#                     p = obstacle.p
#                     self.opti.subject_to(
#                         (self.x[0,k+1]-p[0])**2+(self.x[1,k+1]-p[1])**2>(r+self.quadrotor_r)**2
#                     )
#             # Add Dynamics Constraint
#             self.opti.subject_to(self.x[:, k + 1] == x_next)
#             # Control Constraints (Upper Bound)
#             self.opti.subject_to(self.u[:, k] <= self.upper)
#             # Control Constraints (Lower Bound)
#             self.opti.subject_to(self.u[:, k] >= self.lower)

#         # Setup Optimization Problem
#         self.opti.minimize(self.J)

#         # Setup Covariance Extraction
#         self.CovFuntion = ca.Function('Cov',[self.x, self.u, self.ref,  self.x0, self.X, self.y, self.K], [Cov])

#         opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
#         self.opti.solver(
#             'ipopt', 
#             opts
#         )

#     def __step(self, x, u, wind, dt):
#         """
#         Perform a single RK4 integration step.
#         :param x: Current state
#         :param u: Control input
#         :param dt: Time step
#         :return: State after one RK4 step
#         """
#         k1 = self.dynamics(x, u, wind)
#         k2 = self.dynamics(x + dt / 2 * k1, u, wind)
#         k3 = self.dynamics(x + dt / 2 * k2, u, wind)
#         k4 = self.dynamics(x + dt * k3, u, wind)
#         x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
#         return x_next

#     def __call__(self, x, ref):
#         # Set Initial Guess
#         # Initial guess
#         u_guess = np.repeat(
#             (self.lower[:, np.newaxis] + self.upper[:, np.newaxis]) / 2,
#             self.N,
#             axis=1
#         )
#         x_guess = np.repeat(
#             x[:,np.newaxis],
#             self.N+1,
#             axis=1
#         )
#         self.opti.set_initial(self.x, x_guess)
#         self.opti.set_initial(self.u, u_guess)

#         # Set Initial Condition and Reference
#         self.opti.set_value(self.x0, x)
#         self.opti.set_value(self.ref, ref)

#         # Set GP Data
#         if self.predictor is not None:
#             K, inputs, labels = self.predictor()
#         else:
#             K = np.zeros((self.window_size,self.window_size))
#             inputs = np.zeros((self.input_dim,self.window_size))
#             labels = np.zeros((self.window_size,self.output_dim))
#         self.opti.set_value(self.K, K)
#         self.opti.set_value(self.X, inputs)
#         self.opti.set_value(self.y, labels)

#         # Solve the Problem
#         sol = self.opti.solve()

#         # Extract the solution
#         u_opt = sol.value(self.u)
#         x_opt = sol.value(self.x)

#         # Extract Covariance
#         if self.predictor is not None:
#             Cov = np.array(self.CovFuntion(x_opt,u_opt,ref,x,inputs,labels,K))
#         else:
#             Cov = None
        
#         return np.array(u_opt[:, 0]).flatten(), np.array(x_opt[:2, 1:]), Cov

#     def update_predictor(self, input, label):
#         self.predictor.update(
#             input,
#             label
#         )