import numpy as np
import argparse
import os
import random

from WindField import WindField
from utils.gp_utils import *
from GPModels.ExactGPModels import *

# Argument Parser
# TODO check for argument correctness and maybe add Exceptions
parser = argparse.ArgumentParser()#
parser.add_argument('--save_plots',action='store_true')
parser.add_argument('--show_plots',action='store_true')
parser.add_argument('--test','-t',action='store_true')

# Parse Arguments
args = parser.parse_args()
save_plots = args.save_plots
show_plots = args.show_plots
test = args.test

wind_field = WindField('configs/wind_field.json','configs/mass.json')

# Run the simulation for the different trajectories
for file in os.listdir('trajectories'):
    wind_field.set_trajectory('trajectories/'+file,file)
    wind_field.simulate_wind_field()

# Get GP data
gp_data, x_labels, y_labels, T = wind_field.get_gp_data()

#-----------------------------------------------#
#                Train GP models                #
#-----------------------------------------------#

if not test:
    
    # Exact GP
    idxs = torch.IntTensor(random.sample(range(0,len(gp_data)),200))

    # RBF kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelRBF(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelRBF(train_data,train_y_labels,likelihood_y)
    train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF')

    # RBF + Periodic kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelRBFPeriodic(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelRBFPeriodic(train_data,train_y_labels,likelihood_y)
    train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF-Periodic')

    # Matern 3/2 kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelMatern_32(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelMatern_32(train_data,train_y_labels,likelihood_y)
    train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-32')

    # Matern 5/2 kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelMatern_52(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelMatern_52(train_data,train_y_labels,likelihood_y)
    train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-52')

    # Spectral Mixture (n=4) kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelSpectralMixture_3(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelSpectralMixture_3(train_data,train_y_labels,likelihood_y)
    train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-3')

#----------------------------------------------#
#                Test GP models                #
#----------------------------------------------#

# RBF
test_data = torch.DoubleTensor(gp_data)
test_x_labels = torch.DoubleTensor(x_labels)
test_y_labels = torch.DoubleTensor(y_labels)
likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
model_x = ExactGPModelRBF(test_data,test_x_labels,likelihood_x)
model_y = ExactGPModelRBF(test_data,test_y_labels,likelihood_y)
model_x_dict = torch.load(f'models/ExactGP/model-x-RBF.pth')
model_y_dict = torch.load(f'models/ExactGP/model-y-RBF.pth')
likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-RBF.pth')
likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-RBF.pth')
model_x.load_state_dict(model_x_dict)
model_y.load_state_dict(model_y_dict)
likelihood_x.load_state_dict(likelihood_x_dict)
likelihood_y.load_state_dict(likelihood_y_dict)
test_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF',T,save=save_plots)

# RBF + Periodic
test_data = torch.DoubleTensor(gp_data)
test_x_labels = torch.DoubleTensor(x_labels)
test_y_labels = torch.DoubleTensor(y_labels)
likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
model_x = ExactGPModelRBFPeriodic(test_data,test_x_labels,likelihood_x)
model_y = ExactGPModelRBFPeriodic(test_data,test_y_labels,likelihood_y)
model_x_dict = torch.load(f'models/ExactGP/model-x-RBF-Periodic.pth')
model_y_dict = torch.load(f'models/ExactGP/model-y-RBF-Periodic.pth')
likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-RBF-Periodic.pth')
likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-RBF-Periodic.pth')
model_x.load_state_dict(model_x_dict)
model_y.load_state_dict(model_y_dict)
likelihood_x.load_state_dict(likelihood_x_dict)
likelihood_y.load_state_dict(likelihood_y_dict)
test_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF-Periodic',T,save=save_plots)

# Matern 3/2
test_data = torch.DoubleTensor(gp_data)
test_x_labels = torch.DoubleTensor(x_labels)
test_y_labels = torch.DoubleTensor(y_labels)
likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
model_x = ExactGPModelMatern_32(test_data,test_x_labels,likelihood_x)
model_y = ExactGPModelMatern_32(test_data,test_y_labels,likelihood_y)
model_x_dict = torch.load(f'models/ExactGP/model-x-Matern-32.pth')
model_y_dict = torch.load(f'models/ExactGP/model-y-Matern-32.pth')
likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-Matern-32.pth')
likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-Matern-32.pth')
model_x.load_state_dict(model_x_dict)
model_y.load_state_dict(model_y_dict)
likelihood_x.load_state_dict(likelihood_x_dict)
likelihood_y.load_state_dict(likelihood_y_dict)
test_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-32',T,save=save_plots)

# Matern 5/2
test_data = torch.DoubleTensor(gp_data)
test_x_labels = torch.DoubleTensor(x_labels)
test_y_labels = torch.DoubleTensor(y_labels)
likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
model_x = ExactGPModelMatern_52(test_data,test_x_labels,likelihood_x)
model_y = ExactGPModelMatern_52(test_data,test_y_labels,likelihood_y)
model_x_dict = torch.load(f'models/ExactGP/model-x-Matern-52.pth')
model_y_dict = torch.load(f'models/ExactGP/model-y-Matern-52.pth')
likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-Matern-52.pth')
likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-Matern-52.pth')
model_x.load_state_dict(model_x_dict)
model_y.load_state_dict(model_y_dict)
likelihood_x.load_state_dict(likelihood_x_dict)
likelihood_y.load_state_dict(likelihood_y_dict)
test_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-52',T,save=save_plots)

# Spectral Mixture (n=4)
test_data = torch.DoubleTensor(gp_data)
test_x_labels = torch.DoubleTensor(x_labels)
test_y_labels = torch.DoubleTensor(y_labels)
likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
model_x = ExactGPModelSpectralMixture_3(test_data,test_x_labels,likelihood_x)
model_y = ExactGPModelSpectralMixture_3(test_data,test_y_labels,likelihood_y)
model_x_dict = torch.load(f'models/ExactGP/model-x-SpectralMixture-3.pth')
model_y_dict = torch.load(f'models/ExactGP/model-y-SpectralMixture-3.pth')
likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-SpectralMixture-3.pth')
likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-SpectralMixture-3.pth')
model_x.load_state_dict(model_x_dict)
model_y.load_state_dict(model_y_dict)
likelihood_x.load_state_dict(likelihood_x_dict)
likelihood_y.load_state_dict(likelihood_y_dict)
test_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-3',T,save=save_plots)