import argparse
import json
import torch
import warnings
import os

from pathlib import Path

from WindField import WindField
from utils.train import *
from utils.test_offline import *
from utils.test_online import *
from utils.exceptions import InvalidCommandLineArgumentException

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--save_plots',action='store_true')
parser.add_argument('--show_plots',action='store_true')
parser.add_argument('--test','-t',default=None)
parser.add_argument('--suppress_warnings','-w',action='store_true')

# Parse Arguments
args = parser.parse_args()
save_plots = args.save_plots
show_plots = args.show_plots
suppress_warnings = args.suppress_warnings
test = args.test
if test is not None and test not in ['online','offline','all']:
    raise InvalidCommandLineArgumentException(f'test argument should be among ["online","offline","all"], but "{test}" was provided')

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
    # else "mps" if torch.backends.mps.is_available()
    else "cpu" #use the cpu otherwise
)

wind_field = WindField('configs/wind_field.json','configs/mass.json')

# Run the simulation for the different trajectories
for file in os.listdir('trajectories'):
    wind_field.set_trajectory('trajectories/'+file,file)
    wind_field.simulate_wind_field(show_plots)
    # wind_field.plot(True)
    # wind_field.animate()
    # exit()
    wind_field.reset()

# Get GP data
gp_data, x_labels, y_labels, T = wind_field.get_gp_data()
#-----------------------------------------------#
#                Train GP models                #
#-----------------------------------------------#

if test is None:
    train_ExactGP(gp_data,x_labels,y_labels,exact_gp_options,device,2000)
    train_MultiOutputExactGP(gp_data,x_labels,y_labels,mo_exact_gp_options,device,2000)
    train_SVGP(gp_data,x_labels,y_labels,svgp_options,device,2000)

#------------------------------------------------------#
#                Test GP models Offline                #
#------------------------------------------------------#

if test=="offline" or test=='all':
    wind_field.reset()
    wind_field.reset_gp()
    wind_field.set_trajectory('test_trajectories/lemniscate4.mat','lemniscate4')
    wind_field.simulate_wind_field()
    wind_field.plot(True)
    gp_data, x_labels, y_labels,  = wind_field.get_wind_field_data()
    trajectory_name = Path(file).stem

    test_offline_ExactGP(gp_data,x_labels,y_labels,T,save_plots,exact_gp_options)
    test_offline_MultiOutputExactGP(gp_data,x_labels,y_labels,T,save_plots,mo_exact_gp_options)
    test_offline_SVGP(gp_data,x_labels,y_labels,T,save_plots,svgp_options,'lemniscate4')

#-----------------------------------------------#
#                GP Model Update                #
#-----------------------------------------------#

if test=="online" or test=='all':
    wind_field = WindField('configs/wind_field_test.json','configs/mass.json')
    test_online_svgp(wind_field,'test_trajectories',svgp_options,window_size=50,laps=1)
    test_online_exact_gp(wind_field,'test_trajectories',exact_gp_options,window_size=50,laps=1)
    test_online_exact_mogp(wind_field,'test_trajectories',mo_exact_gp_options,50,laps=1,horizon=1)