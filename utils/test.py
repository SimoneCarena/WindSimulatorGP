import gpytorch
import torch
import os
import pickle

from pathlib import Path

from GPModels.ExactGPModel import *
from GPModels.SVGPModel import *
from GPModels.MultiOutputExactGPModel import *

from modules.GPModel import GPModel
from modules.Kernels import RBFKernel

from modules.MPC import *

@torch.no_grad
def __test_svgp(wind_field, trajectories_folder, model_x, model_y, name, window_size, p0, laps, show, save):

    # Put the models in eval mode
    model_x.eval()
    model_y.eval()
    # Start By resetting the wind field
    wind_field.reset()
    wind_field.reset_gp()
    wind_field.draw_wind_field(True)

    # Setup Predictor for the simulation
    kernel = RBFKernel(
            model_x.covar_module.base_kernel.lengthscale.item(),
            model_x.covar_module.data_covar_module.outputscale.item(),
        )
    predictor = GPModel(
        kernel,
        0.001,
        2,
        3,
        window_size,
    )

    for file in os.listdir(trajectories_folder):
        file_name = Path(file).stem
        wind_field.set_trajectory(trajectories_folder+'/'+file,file_name,laps)
        print('==========================================')
        print('  Simulating Wind Field without baseline  ')
        print('==========================================')
        wind_field.simulate_wind_field(False,show)
        wind_field.reset()
        wind_field.reset_gp()
        print('=======================================')
        print('  Simulating Wind Field with baseline  ')
        print('=======================================')
        wind_field.simulate_wind_field(True,show)
        wind_field.reset()
        wind_field.reset_gp()
        print('=================================')
        print('  Simulating Wind Field with gp  ')
        print('=================================')
        wind_field.simulate_mogp(window_size,predictor,p0,show=show,save=save,kernel_name=name) 
        wind_field.reset()
        wind_field.reset_gp()
        print('Done')

@torch.no_grad
def __test_exact_gp(wind_field, trajectories_folder, model_x, model_y, name, window_size, p0, laps, show, save):

    # Put the models in eval mode
    model_x.eval()
    model_y.eval()
    # Start By resetting the wind field
    wind_field.reset()
    wind_field.reset_gp()
    wind_field.draw_wind_field(True)

    # Setup Predictor for the simulation
    kernel = RBFKernel(
            model_x.covar_module.base_kernel.lengthscale.item(),
            model_x.covar_module.data_covar_module.outputscale.item(),
        )
    predictor = GPModel(
        kernel,
        0.001,
        2,
        3,
        window_size,
    )

    for file in os.listdir(trajectories_folder):
        file_name = Path(file).stem
        wind_field.set_trajectory(trajectories_folder+'/'+file,file_name,laps)
        print('==========================================')
        print('  Simulating Wind Field without baseline  ')
        print('==========================================')
        wind_field.simulate_wind_field(False,show)
        wind_field.reset()
        wind_field.reset_gp()
        print('=======================================')
        print('  Simulating Wind Field with baseline  ')
        print('=======================================')
        wind_field.simulate_wind_field(True,show)
        wind_field.reset()
        wind_field.reset_gp()
        print('=================================')
        print('  Simulating Wind Field with gp  ')
        print('=================================')
        wind_field.simulate_mogp(window_size,predictor,p0,show=show,save=save,kernel_name=name) 
        wind_field.reset()
        wind_field.reset_gp()
        print('Done')

@torch.no_grad
def __test_exact_mogp(wind_field, trajectories_folder, model, name, window_size, p0, laps, show, save):

    # Put the models in eval mode
    model.eval()
    # Start By resetting the wind field
    wind_field.reset()
    wind_field.reset_gp()
    wind_field.draw_wind_field(show,save)

    # Setup Predictor for the simulation
    kernel = RBFKernel(
            model.covar_module.data_covar_module.base_kernel.lengthscale.item(),
            model.covar_module.data_covar_module.outputscale.item(),
        )
    predictor = GPModel(
        kernel,
        model.likelihood.noise.item(),
        2,
        3,
        window_size,
    )

    # print(model.covar_module.data_covar_module.base_kernel.lengthscale.item())
    # print(model.covar_module.data_covar_module.outputscale.item())
    # print(model.likelihood.noise.item()*10**4)
    # exit()
    
    # Create Log Files
    log_files = []
    rmse_log = open("rmse.csv","w")
    time_log = open("solver_time.csv","w")
    log_files.append(rmse_log)
    log_files.append(time_log)

    rmse_log.write("No-Wind, Baseline, GP (acados), GP (CasADi), Trajectory\n")
    time_log.write("No-Wind, Baseline, GP (acados), GP (CasADi)\n")

    for file in sorted(os.listdir(trajectories_folder)):
        file_name = Path(file).stem
        wind_field.set_trajectory(trajectories_folder+'/'+file,file_name,laps)
        print('==========================================')
        print('  Simulating Wind Field without baseline  ')
        print('==========================================')
        wind_field.simulate_wind_field(False,show,save,log_files)
        wind_field.reset()
        wind_field.reset_gp()
        rmse_log.write(", ")
        time_log.write(", ")
        print('=======================================')
        print('  Simulating Wind Field with baseline  ')
        print('=======================================')
        wind_field.simulate_wind_field(True,show,save,log_files)
        wind_field.reset()
        wind_field.reset_gp()
        rmse_log.write(", ")
        time_log.write(", ")
        print('==========================================')
        print('  Simulating Wind Field with gp (acados)  ')
        print('==========================================')
        wind_field.simulate_mogp(window_size,predictor,p0,show=show,save=save, log_files = log_files, mpc_type = MPCAcados) 
        wind_field.reset()
        wind_field.reset_gp()
        rmse_log.write(", ")
        time_log.write(", ")
        print('==========================================')
        print('  Simulating Wind Field with gp (casadi)  ')
        print('==========================================')
        wind_field.simulate_mogp(window_size,predictor,p0,show=show,save=save, log_files = log_files, mpc_type = MPCIpopt) 
        wind_field.reset()
        wind_field.reset_gp()
        rmse_log.write(f", {file_name}\n")
        time_log.write("\n")

    rmse_log.close()
    time_log.close()

def test_svgp(wind_field, trajecotries_folder, options, window_size=100, p0=None, laps=1, show=False, save=None):
    file = open(".metadata/svgp_dict","rb")
    svgp_dict = pickle.load(file)
    file.close()
    file = open(".metadata/exact_gp_dict","rb")
    exact_gp_dict = pickle.load(file)
    file.close()

    for name in svgp_dict:
        if options[name]:
            likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
            likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
            likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-{name}.pth')
            likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-{name}.pth')
            likelihood_x.load_state_dict(likelihood_x_dict)
            likelihood_y.load_state_dict(likelihood_y_dict)
            inducing_points_x = torch.load(f'data/SVGP/inducing_points_x-{name}.pt')
            inducing_points_y = torch.load(f'data/SVGP/inducing_points_y-{name}.pt')
            model_x = svgp_dict[name](inducing_points_x)
            model_y = svgp_dict[name](inducing_points_y)
            model_x_dict = torch.load(f'models/SVGP/model-x-{name}.pth')
            model_y_dict = torch.load(f'models/SVGP/model-y-{name}.pth')
            model_x.load_state_dict(model_x_dict)
            model_y.load_state_dict(model_y_dict)

            # Retrieve Exact GP x
            model = exact_gp_dict[name](torch.empty((0,2)),torch.empty((0,1)),likelihood_x)
            model.covar_module = model_x.covar_module
            model_x = model

            # Retrieve Exact GP y
            model = exact_gp_dict[name](torch.empty((0,2)),torch.empty((0,1)),likelihood_y)
            model.covar_module = model_y.covar_module
            model_y = model

            __test_svgp(wind_field,trajecotries_folder,model_x,model_y,name,window_size,laps,show,save)

def test_exact_gp(wind_field, trajecotries_folder, options, window_size=100, p0=None, laps=1, show=False, save=None):
    file = open(".metadata/exact_gp_dict","rb")
    exact_gp_dict = pickle.load(file)

    for name in exact_gp_dict:
        if options[name]:
            likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
            likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
            likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-{name}.pth')
            likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-{name}.pth')
            likelihood_x.load_state_dict(likelihood_x_dict)
            likelihood_y.load_state_dict(likelihood_y_dict)
            model_x = exact_gp_dict[name](torch.empty((0,2)),torch.empty((0,1)),likelihood_x)
            model_y = exact_gp_dict[name](torch.empty((0,2)),torch.empty((0,1)),likelihood_y)
            model_x_dict = torch.load(f'models/ExactGP/model-x-{name}.pth')
            model_y_dict = torch.load(f'models/ExactGP/model-y-{name}.pth')
            model_x.load_state_dict(model_x_dict)
            model_y.load_state_dict(model_y_dict)
            __test_exact_gp(wind_field,trajecotries_folder,model_x,model_y,name,window_size,p0,laps,show,save)

def test_exact_mogp(wind_field, trajecotries_folder, options, window_size=100, p0=None, laps=1, show=False, save=None):
    file = open(".metadata/mo_exact_gp_dict","rb")
    mo_exact_gp_dict = pickle.load(file)

    for name in mo_exact_gp_dict:
        if options[name]:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=2,
                rank=0,
                has_global_noise = True,
                has_task_noise = False
            )
            likelihood_dict = torch.load(f'models/MultiOutputExactGP/likelihood-{name}.pth')
            likelihood.load_state_dict(likelihood_dict)
            model = mo_exact_gp_dict[name](torch.empty((0,2)),torch.empty((0,2)),likelihood)
            model_dict = torch.load(f'models/MultiOutputExactGP/model-{name}.pth')
            model.load_state_dict(model_dict)

            __test_exact_mogp(
                wind_field,
                trajecotries_folder,
                model,
                name,
                window_size,
                p0,
                laps,
                show,
                save
            )