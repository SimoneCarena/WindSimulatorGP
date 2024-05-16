import gpytorch
import torch
import matplotlib.pyplot as plt
import numpy as np

from GPModels.ExactGPModels import *
from GPModels.MultiOutputExactGPModels import *
from GPModels.SVGPModels import *

@torch.no_grad
def __test_online_gp():
    pass

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