def get_acados_mhe_solver(model, M, ts):
    # Obtain number of states and inputs
    nx = model.get_n_states()
    nu = model.get_n_inputs()
    ny = model.get_n_outputs()
    nw = model.get_n_mismatch()

    # Define shooting variables
    x_est = SX.sym("x", nx)
    w_est = SX.sym("w", nw)

    # Define parameters
    u_applied = SX.sym("u_applied", nu)
    y_meas = SX.sym("y_meas", ny)

    # Create OCP dimensions
    ocp_dims = AcadosOcpDims()
    ocp_dims.N = M

    # Create acados model
    acados_model = AcadosModel()
    acados_model.name = model.get_name()
    acados_model.x = x_est
    acados_model.u = vertcat(w_est)
    acados_model.p = vertcat(u_applied, y_meas)
    # state update equality constraint
    acados_model.f_expl_expr = model.state_update_ct_noise(x_est, u_applied, w_est)
    # symbolic cost terms
    acados_model.cost_y_expr_0 = vertcat(w_est, x_est - y_meas)
    acados_model.cost_y_expr = vertcat(w_est, x_est - y_meas)
    acados_model.cost_y_expr_e = x_est - y_meas

    # Create OCP cost
    ocp_cost = AcadosOcpCost()
    ocp_cost.cost_type_0 = "NONLINEAR_LS"
    ocp_cost.W_0 = np.zeros((nw + ny, nw + ny))
    ocp_cost.yref_0 = np.zeros((nw + ny,))
    ocp_cost.cost_type = "NONLINEAR_LS"
    ocp_cost.W = np.zeros((nw + ny, nw + ny))
    ocp_cost.yref = np.zeros((nw + ny,))
    ocp_cost.cost_type_e = "NONLINEAR_LS"
    ocp_cost.W_e = np.zeros((ny, ny))
    ocp_cost.yref_e = np.zeros((ny,))

    # Create OCP options object
    ocp_options = AcadosOcpOptions()
    ocp_options.tf = M * ts
    ocp_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp_options.hessian_approx = "GAUSS_NEWTON"
    ocp_options.integrator_type = "ERK"
    ocp_options.nlp_solver_type = "SQP"

    # Create OCP object
    # NOTE: no constraints are put on the shooting variables
    ocp = AcadosOcp()
    ocp.dims = ocp_dims
    ocp.model = acados_model
    ocp.cost = ocp_cost
    ocp.solver_options = ocp_options
    ocp.parameter_values = np.zeros((acados_model.p.rows(),))

    # Create OCP solver
    return AcadosOcpSolver(ocp, json_file=f"acados_mhe_{M}.json")