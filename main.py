import argparse
import json
import torch
import warnings
import os

from WindField import WindField
from utils.train import *
from utils.test import *

# Argument Parser
# TODO check for argument correctness and maybe add Exceptions
parser = argparse.ArgumentParser()
parser.add_argument('--save_plots',action='store_true')
parser.add_argument('--show_plots',action='store_true')
parser.add_argument('--test','-t',action='store_true')
parser.add_argument('--suppress_warnings','-w',action='store_true')

# Parse Arguments
args = parser.parse_args()
save_plots = args.save_plots
show_plots = args.show_plots
test = args.test
suppress_warnings = args.suppress_warnings

# Suppress Warning (caused by GPytorch internal stuff)
if suppress_warnings:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

# Parse GP Options
## Exact GP
file = open('configs/exact_gp.json')
exact_gp_options = json.load(file)
file.close()
## MultiOutput ExactGP
file = open('configs/mo_exact_gp.json')
mo_exact_gp_options = json.load(file)
file.close()
## SVGP
file = open('configs/svgp.json')
svgp_options = json.load(file)
file.close()

# Setup the Device for training
device = (
    "cuda"
    if torch.cuda.is_available() #Use cuda if available
    else "cpu" #use the cpu otherwise
)

wind_field = WindField('configs/wind_field.json','configs/mass.json')

# Run the simulation for the different trajectories
for file in os.listdir('trajectories'):
    wind_field.set_trajectory('trajectories/'+file,file)
    wind_field.simulate_wind_field()
    if show_plots:  
        wind_field.plot(save_plots)
    wind_field.reset()


# Get GP data
gp_data, x_labels, y_labels, T = wind_field.get_gp_data()
#-----------------------------------------------#
#                Train GP models                #
#-----------------------------------------------#

if not test:
    # train_ExactGP(gp_data,x_labels,y_labels,exact_gp_options,device,10000)
    # train_MultiOutputExactGP(gp_data,x_labels,y_labels,mo_exact_gp_options,device,20000)
    train_SVGP(gp_data,x_labels,y_labels,svgp_options,device,20000)

#----------------------------------------------#
#                Test GP models                #
#----------------------------------------------#

wind_field.reset()
wind_field.reset_gp()
wind_field.set_trajectory('trajectories/lemniscate.mat','lemniscate.mat')
wind_field.simulate_wind_field()
gp_data, x_labels, y_labels, T = wind_field.get_gp_data()
wind_field_data = wind_field.get_wind_field_data()

# test_ExactGP(gp_data,x_labels,y_labels,T,save_plots,exact_gp_options)
# test_MultiOutputExactGP(gp_data,x_labels,y_labels,T,save_plots,mo_exact_gp_options)
test_SVGP(gp_data,x_labels,y_labels,T,save_plots,svgp_options)

# likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
# likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
# likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-RBF.pth')
# likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-RBF.pth')
# likelihood_x.load_state_dict(likelihood_x_dict)
# likelihood_y.load_state_dict(likelihood_y_dict)
# inducing_points = torch.load('data/SVGP/inducing_points-RBF.pt')
# model_x = SVGPModelRBF(inducing_points)
# model_y = SVGPModelRBF(inducing_points)
# model_x_dict = torch.load(f'models/SVGP/model-x-RBF.pth')
# model_y_dict = torch.load(f'models/SVGP/model-y-RBF.pth')
# model_x.load_state_dict(model_x_dict)
# model_y.load_state_dict(model_y_dict)

# wind_field = WindField('configs/wind_field.json','configs/mass.json',model_x,model_y)
# wind_field.set_trajectory('trajectories/lemniscate.mat','lemniscate.mat')
# wind_field.simulate_wind_field()  
# wind_field.plot()
# wind_field.reset()