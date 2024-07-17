import numpy as np

from acados_template import AcadosOcp, AcadosOcpSolver

class MPC:
    def __init__(self, drone_dynamics, horizon, Q, R):
        self.drone_dynamics = drone_dynamics
        self.horizon = horizon
        self.Q = Q
        self.R = R
        self.ocp_solver = self.__create_acados_ocp()

    def __create_acados_ocs(self):
        pass