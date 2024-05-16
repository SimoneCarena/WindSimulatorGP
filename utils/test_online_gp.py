import gpytorch
import torch
import os

from pathlib import Path

from GPModels.ExactGPModels import *
from GPModels.SVGPModels import *

@torch.no_grad
def __test_online_gp(wind_field, trajectories_folder, model_x, model_y, name):

    # Start By resetting the wind field
    wind_field.reset(gp_predictor_x=model_x, gp_predictor_y=model_y)
    wind_field.reset_gp()
    # Put the models in eval mode
    model_x.eval()
    model_y.eval()

    for file in os.listdir(trajectories_folder):
        file_name = Path(file).stem
        wind_field.set_trajectory(trajectories_folder+'/'+file,file_name)
        wind_field.simulate_one_step_gp(50,show=True,save='imgs/gp_update_plots/'+name+'-'+file_name,kernel_name=name)  
        wind_field.reset()
        wind_field.reset_gp()


def test_online_gp(wind_field, trajecotries_folder, options):
    # RBF
    if options['RBF'] == True:
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-RBF.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-RBF.pth')
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        inducing_points_x = torch.load('data/SVGP/inducing_points_x-RBF.pt')
        inducing_points_y = torch.load('data/SVGP/inducing_points_y-RBF.pt')
        model_x = SVGPModelRBF(inducing_points_x)
        model_y = SVGPModelRBF(inducing_points_y)
        model_x_dict = torch.load(f'models/SVGP/model-x-RBF.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-RBF.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)

        # Retrieve Exact GP x
        model = ExactGPModelRBF(torch.empty((0,2)),torch.empty((0,1)),likelihood_x)
        model.covar_module = model_x.covar_module
        model_x = model

        # Retrieve Exact GP y
        model = ExactGPModelRBF(torch.empty((0,2)),torch.empty((0,1)),likelihood_y)
        model.covar_module = model_y.covar_module
        model_y = model

        __test_online_gp(wind_field,trajecotries_folder,model_x,model_y,'RBF')

    # RBF + Periodic
    if options['RBF-Periodic'] == True:
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-RBF-Periodic.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-RBF-Periodic.pth')
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        inducing_points_x = torch.load('data/SVGP/inducing_points_x-RBF-Periodic.pt')
        inducing_points_y = torch.load('data/SVGP/inducing_points_y-RBF-Periodic.pt')
        model_x = SVGPModelRBFPeriodic(inducing_points_x)
        model_y = SVGPModelRBFPeriodic(inducing_points_y)
        model_x_dict = torch.load(f'models/SVGP/model-x-RBF-Periodic.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-RBF-Periodic.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)

        # Retrieve Exact GP x
        model = ExactGPModelRBFPeriodic(torch.empty((0,2)),torch.empty((0,1)),likelihood_x)
        model.covar_module = model_x.covar_module
        model_x = model

        # Retrieve Exact GP y
        model = ExactGPModelRBFPeriodic(torch.empty((0,2)),torch.empty((0,1)),likelihood_y)
        model.covar_module = model_y.covar_module
        model_y = model

        __test_online_gp(wind_field,trajecotries_folder,model_x,model_y,'RBF-Periodic')

    # Matern-32
    if options['Matern-32'] == True:
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-Matern-32.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-Matern-32.pth')
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        inducing_points_x = torch.load('data/SVGP/inducing_points_x-Matern-32.pt')
        inducing_points_y = torch.load('data/SVGP/inducing_points_y-Matern-32.pt')
        model_x = SVGPModelMatern_32(inducing_points_x)
        model_y = SVGPModelMatern_32(inducing_points_y)
        model_x_dict = torch.load(f'models/SVGP/model-x-Matern-32.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-Matern-32.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)

        # Retrieve Exact GP x
        model = ExactGPModelMatern_32(torch.empty((0,2)),torch.empty((0,1)),likelihood_x)
        model.covar_module = model_x.covar_module
        model_x = model

        # Retrieve Exact GP y
        model = ExactGPModelMatern_32(torch.empty((0,2)),torch.empty((0,1)),likelihood_y)
        model.covar_module = model_y.covar_module
        model_y = model

        __test_online_gp(wind_field,trajecotries_folder,model_x,model_y,'Matern-32')

    # Matern-52
    if options['Matern-52'] == True:
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-Matern-52.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-Matern-52.pth')
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        inducing_points_x = torch.load('data/SVGP/inducing_points_x-Matern-52.pt')
        inducing_points_y = torch.load('data/SVGP/inducing_points_y-Matern-52.pt')
        model_x = SVGPModelMatern_52(inducing_points_x)
        model_y = SVGPModelMatern_52(inducing_points_y)
        model_x_dict = torch.load(f'models/SVGP/model-x-Matern-52.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-Matern-52.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)

        # Retrieve Exact GP x
        model = ExactGPModelMatern_52(torch.empty((0,2)),torch.empty((0,1)),likelihood_x)
        model.covar_module = model_x.covar_module
        model_x = model

        # Retrieve Exact GP y
        model = ExactGPModelMatern_52(torch.empty((0,2)),torch.empty((0,1)),likelihood_y)
        model.covar_module = model_y.covar_module
        model_y = model

        __test_online_gp(wind_field,trajecotries_folder,model_x,model_y,'Matern-52')

    # SpectralMixture-3
    if options['SpectralMixture-3'] == True:
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-SpectralMixture-3.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-SpectralMixture-3.pth')
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        inducing_points_x = torch.load('data/SVGP/inducing_points_x-SpectralMixture-3.pt')
        inducing_points_y = torch.load('data/SVGP/inducing_points_y-SpectralMixture-3.pt')
        model_x = SVGPModelSpectralMixture_3(inducing_points_x)
        model_y = SVGPModelSpectralMixture_3(inducing_points_y)
        model_x_dict = torch.load(f'models/SVGP/model-x-SpectralMixture-3.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-SpectralMixture-3.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)

        # Retrieve Exact GP x
        model = ExactGPModelSpectralMixture_3(torch.empty((0,2)),torch.empty((0,1)),likelihood_x)
        model.covar_module = model_x.covar_module
        model_x = model

        # Retrieve Exact GP y
        model = ExactGPModelSpectralMixture_3(torch.empty((0,2)),torch.empty((0,1)),likelihood_y)
        model.covar_module = model_y.covar_module
        model_y = model

        __test_online_gp(wind_field,trajecotries_folder,model_x,model_y,'SpectralMixture-3')

    # SpectralMixture-5
    if options['SpectralMixture-5'] == True:
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-SpectralMixture-5.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-SpectralMixture-5.pth')
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        inducing_points_x = torch.load('data/SVGP/inducing_points_x-SpectralMixture-5.pt')
        inducing_points_y = torch.load('data/SVGP/inducing_points_y-SpectralMixture-5.pt')
        model_x = SVGPModelSpectralMixture_5(inducing_points_x)
        model_y = SVGPModelSpectralMixture_5(inducing_points_y)
        model_x_dict = torch.load(f'models/SVGP/model-x-SpectralMixture-5.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-SpectralMixture-5.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)

        # Retrieve Exact GP x
        model = ExactGPModelSpectralMixture_5(torch.empty((0,2)),torch.empty((0,1)),likelihood_x)
        model.covar_module = model_x.covar_module
        model_x = model

        # Retrieve Exact GP y
        model = ExactGPModelSpectralMixture_5(torch.empty((0,2)),torch.empty((0,1)),likelihood_y)
        model.covar_module = model_y.covar_module
        model_y = model

        __test_online_gp(wind_field,trajecotries_folder,model_x,model_y,'SpectralMixture-5')

    # SpectralMixture-10
    if options['SpectralMixture-10'] == True:
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-SpectralMixture-10.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-SpectralMixture-10.pth')
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        inducing_points_x = torch.load('data/SVGP/inducing_points_x-SpectralMixture-10.pt')
        inducing_points_y = torch.load('data/SVGP/inducing_points_y-SpectralMixture-10.pt')
        model_x = SVGPModelSpectralMixture_10(inducing_points_x)
        model_y = SVGPModelSpectralMixture_10(inducing_points_y)
        model_x_dict = torch.load(f'models/SVGP/model-x-SpectralMixture-10.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-SpectralMixture-10.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)

        # Retrieve Exact GP x
        model = ExactGPModelSpectralMixture_10(torch.empty((0,2)),torch.empty((0,1)),likelihood_x)
        model.covar_module = model_x.covar_module
        model_x = model

        # Retrieve Exact GP y
        model = ExactGPModelSpectralMixture_10(torch.empty((0,2)),torch.empty((0,1)),likelihood_y)
        model.covar_module = model_y.covar_module
        model_y = model

        __test_online_gp(wind_field,trajecotries_folder,model_x,model_y,'SpectralMixture-10')