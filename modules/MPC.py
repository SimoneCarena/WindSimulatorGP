import numpy as np
import casadi as ca
import sys

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import vertcat
from scipy.stats import chi2

class MPC:
    def __init__(self, model, control_horizon, dt, Q, R, maximum_solver_time ,predictor=None, obstacles=[]):
        # Quadrotor data
        self.model = model
        self.dynamics = model.get_dynamics()
        self.diff_dynamics = model.get_diff_dynamics()
        self.quadrotor_r = model.r
        self.nx, self.nu, self.ny = model.get_dimensions()

        # GP prediction data
        if predictor is not None:
            self.predictor = predictor
            self.gp_on = False
            self.window_size, self.input_dim, self.output_dim = predictor.get_dims()

        # Control data
        self.Q = Q
        self.R = R
        self.N = control_horizon
        self.dt = dt
        self.obstacles = obstacles
        self.lower = np.array([-np.pi/6, -np.pi/6, -np.pi/6, 5])
        self.upper = np.array([np.pi/6, np.pi/6, np.pi/6, 15])
        
        # Uncertainty data
        self.delta = 0.05
        self.chi2_val = np.sqrt(chi2.ppf(1-self.delta, 2))

        # Setupd the solver(s)
        self.__setup_solver()
        if predictor is not None:
            self.__setup_solver_gp() 
            self.__setup_gp_prediction()
        # Set the default solver to be the one without gp
        self.solver = self.solver_no_gp

        # Maximum time allowed for the solver to solve the optimization problem
        self.max_time = maximum_solver_time

    def __setup_solver(self):
        # Declare control variables
        x = ca.SX.sym('x',self.nx)
        u = ca.SX.sym('u',self.nu)

        # Create acados ocp probelm
        ocp = AcadosOcp()

        # Use the previously created acados model
        model = AcadosModel()
        model.x = x
        model.u = u
        model.f_expl_expr = self.dynamics(x,u,ca.SX.zeros(3))
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
        ocp.cost.yref = np.zeros(self.ny+self.nu)
        ## The terminal cost only referencec the state
        ocp.cost.yref_e = np.zeros(self.ny)

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
                    (x[:2]-p0).T@(x[:2]-p0)-(r+self.quadrotor_r)**2
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
        self.solver_no_gp = AcadosOcpSolver(ocp, json_file='acados/acados_ocp.json')

    def __setup_solver_gp(self):
        # Declare control variables
        x = ca.SX.sym('x',self.nx)
        u = ca.SX.sym('u',self.nu)
        # Mean of gp wind prediction
        mean = ca.SX.sym('mean',3)
        # Covariance in the gp position
        cov = ca.SX.sym('cov',2,2)

        # Create the ocp problem
        ocp = AcadosOcp()

        # Use the previously created acados model
        model = AcadosModel()
        model.x = x
        model.u = u
        model.f_expl_expr = self.dynamics(x,u,mean)
        model.p = ca.vertcat(
            mean.reshape((-1,1)),
            cov.reshape((-1,1))
        )
        model.name = 'quadrotor_gp_mpc'

        ocp.model = model
        ocp.parameter_values = np.zeros((
            3 + 2*2,
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
        ocp.cost.yref = np.zeros(self.ny+self.nu)
        ## The terminal cost only referencec the state
        ocp.cost.yref_e = np.zeros(self.ny)

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
            # Inequality constraints are expressed in the form
            # lh <= h(x,u) <= uh, the initial constraint is expressed separately
            # but with the same formulation lh_0 <= h(x,u) <= uh_0
            h = []
            lh_0 = []
            uh_0 = []
            lh = []
            uh = []
            # Length of the uncertainty ellipsoid
            lx = ca.sqrt(cov[0,0])*self.chi2_val
            ly = ca.sqrt(cov[1,1])*self.chi2_val
            l = ca.fmax(lx,ly)
            for obstacle in self.obstacles:
                # Constraints are of the form
                # 0 <= dist(quadrotor,obstacle) - (q_r+r+l)^2 <= inf
                r = obstacle.r
                p0 = obstacle.p
                d2 = (r+self.quadrotor_r+l)**2
                h.append(
                    (x[:2]-p0).T@(x[:2,:]-p0)-d2
                )
                lh_0.append(0)
                lh.append(0)
                uh_0.append(1000)
                uh.append(1000)
            ocp.model.con_h_expr_0 = ca.vertcat(*h)
            ocp.model.con_h_expr = ca.vertcat(*h)
            ocp.constraints.lh_0 = np.array(lh_0)
            ocp.constraints.uh_0 =  np.array(uh_0)
            ocp.constraints.lh =  np.array(lh)
            ocp.constraints.uh =  np.array(uh)

        # Create the solver
        ocp.code_export_directory = 'acados/solver_gp'
        self.solver_gp = AcadosOcpSolver(ocp, json_file='acados/acados_ocp.json')

    def __setup_gp_prediction(self):
        # Define GP variables
        K_inv = ca.SX.sym("K_inv",self.window_size,self.window_size)    # Inverse of the kernel matrix
        X = ca.SX.sym("X",self.input_dim,self.window_size)              # Inputs for the model
        y = ca.SX.sym("y",self.window_size,self.output_dim)             # Ouptus for the model  
        cov_x_0 = ca.SX.sym("cov_x_0",2,2)                              # Covariance matrix on the position
        x = ca.SX.sym("x",self.nx)                                      # State vector

        # Compute predictive mean and variance
        K_xx = []
        for t in range(self.window_size):
            K_xx.append(
                self.predictor.kernel(
                    x[:self.input_dim],
                    X[:,t]
                )
            )
        K_xx = ca.vertcat(*K_xx)
        mean = K_xx.T@K_inv@y
        cov_gp = self.predictor.kernel(x[:self.input_dim],x[:self.input_dim])-K_xx.T@K_inv@K_xx
        cov_gp = cov_gp*np.eye(self.input_dim)

        # Propagate the uncertainty for the position on the next state
        A = self.diff_dynamics(x,mean)
        A = A[:2,:2]
        K_xx_d = []
        for j in range(self.window_size):
            K_xx_d.append(self.predictor.kernel_derivative(x[:self.input_dim],X[:,j]).T)
        K_xx_d = ca.vertcat(*K_xx_d)
        mean_d = K_xx_d.T@K_inv@y[:,:2]
        Sigma_xd = mean_d@cov_x_0
        _first_mat = ca.horzcat(A,np.eye(2))
        _second_mat = ca.vertcat(
            ca.horzcat(cov_x_0, Sigma_xd),
            ca.horzcat(Sigma_xd.T, cov_gp)
        )
        _third_mat = _first_mat.T
        cov_x = _first_mat@ _second_mat@ _third_mat

        # Create the function objects to compute the predictive mean 
        # and the uncertainty on the next state
        self.predictive_mean = ca.Function(
            "mu",
            [x,K_inv,X,y],
            [mean]
        )
        self.propagate_uncertainty = ca.Function(
            "cov_x",
            [x,K_inv,X,y,cov_x_0],
            [cov_x]
        )

    def __call__(self, x, ref, prev_x_opt):
        """
        Solve the MPC optimization problem.

        :param x: The current state (initial condition), shape (nx,).
        :param ref: The reference trajectory, shape (nx, N+1).
        :param prev_x_opt: The previously optimized states over the horizon N
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

        # If the GP prediction is set, use the model to make the wind predictions
        if self.gp_on:
            K_inv, X, y = self.predictor()
            cov_x = np.zeros((2,2))
            Covs = []
            for k in range(self.N):
                # Compute wind prediction
                mean = self.predictive_mean(prev_x_opt[:,k],K_inv,X,y).full()
                params = np.concatenate([
                    mean.reshape((-1,1)),
                    cov_x.reshape((-1,1))
                ])
                self.solver.set(
                    self.N,
                    "p",
                    params
                )
                # Propagate the uncertainty on the position
                cov_x = self.propagate_uncertainty(prev_x_opt[:,k],K_inv,X,y,cov_x).full()
                Covs.append(cov_x.copy())
            Covs = np.vstack(Covs)

        # Solve the optimization problem
        status = self.solver.solve()

        # Check the solver status
        if status != 0:
            print(
                '\033[93m'+f"Solver Returned Exit Status {4}"+'\033[93m',
                file = sys.stderr
            )
        # Check the execution time and verify it is under the control time
        solver_time = self.solver.get_stats('time_tot')
        if solver_time > self.max_time:
            print(
                """\033[93m
                Solver exceeded maximum computation time\n
                Solver time: {:.2f} s\n
                Maximum control time: 0.1 s\n
                \033[93m""".format(solver_time),
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
        if not self.gp_on:
            return u_opt[0, :], x_opt.T, None
        else:
            return u_opt[0, :], x_opt.T, Covs
    
    def update_predictor(self, input, label):
        '''
        Updates the predictor used by the mpc to estimate the wind
        '''
        self.predictor.update(
            input,
            label
        )

    def set_predictor(self):
        '''
        Enables the prediction of the wind using the gp model, and the 
        chance-constrained obstacle avoidance using the variance of the gp model
        '''
        self.gp_on = True
        self.solver = self.solver_gp


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