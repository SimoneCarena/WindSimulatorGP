import argparse
import json
import torch
import warnings
import os
import numpy as np

from WindField import WindField
from utils.train import *
from utils.test import *

from utils.exceptions import InvalidCommandLineArgumentException

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--save_plots',default='None')
parser.add_argument('--show_plots',default='None')
parser.add_argument('--test','-t',action='store_true')
parser.add_argument('--suppress_warnings','-w',action='store_true')

# Parse Arguments
args = parser.parse_args()
save_plots = args.save_plots
show_plots = args.show_plots
suppress_warnings = args.suppress_warnings
test = args.test

permitted_arguments = ['train', 'test', 'all', 'None']

if show_plots not in permitted_arguments:
    raise InvalidCommandLineArgumentException(f'{show_plots} is not among allowed "show_plots" arguments ({permitted_arguments})')
if save_plots not in permitted_arguments:
    raise InvalidCommandLineArgumentException(f'{save_plots} is not among allowed "savle_plots" arguments ({permitted_arguments})')

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
    if torch.cuda.is_available() # Use cuda if available
    # else "mps" if torch.backends.mps.is_available() # Unfortunately it is not supported :(
    else "cpu" # Use the cpu otherwise
)

# Train Models
if not test:
    wind_field = WindField('configs/wind_field.json')

    # Run the simulation for the different trajectories
    for file in os.listdir('trajectories'):
        wind_field.set_trajectory('trajectories/'+file,file)
        wind_field.simulate_wind_field()
        wind_field.draw_wind_field(
            show=True if (show_plots == 'train' or show_plots == 'all') else False,
            save='imgs/wind_field' if (save_plots == 'train' or save_plots == 'all') else None
        )
        wind_field.plot(
            show=True if (show_plots == 'train' or show_plots == 'all') else False,
            save='imgs/wind_field' if (save_plots == 'train' or save_plots == 'all') else None
        )
        wind_field.reset()

    # Get GP data
    gp_data, x_labels, y_labels = wind_field.get_gp_data()

    # Train the models
    train_ExactGP(gp_data,x_labels,y_labels,exact_gp_options,device,2000)
    train_MultiOutputExactGP(gp_data,x_labels,y_labels,mo_exact_gp_options,device,20000)
    train_SVGP(gp_data,x_labels,y_labels,svgp_options,device,2000)
# Test Models
else:
    wind_field = WindField('configs/wind_field_test.json')
    # test_svgp(
    #     wind_field,
    #     'test_trajectories',
    #     svgp_options,
    #     window_size = 50,
    #     p0 = np.array([4.0,2.0]),
    #     laps = 1,
    #     show = True if (show_plots == 'test' or show_plots == 'all') else False,
    #     save = 'imgs/gp_updates/SVGP/' if (save_plots == 'test' or save_plots == 'all') else None
    # )
    # test_exact_gp(
    #     wind_field,
    #     'test_trajectories',
    #     exact_gp_options,
    #     window_size = 50,
    #     p0 = np.array([4.0,2.0]),
    #     laps = 1,
    #     horizon = 100,
    #     show = True if (show_plots == 'test' or show_plots == 'all') else False,
    #     save = 'imgs/gp_updates/ExactGP/' if (save_plots == 'test' or save_plots == 'all') else None
    # )
    test_exact_mogp(
        wind_field,
        'test_trajectories',
        mo_exact_gp_options,
        window_size = 20,
        laps = 1,
        # p0=np.array([3,1,0.5]),
        horizon = 1,
        show = True if (show_plots == 'test' or show_plots == 'all') else False,
        save = 'imgs/gp_updates/MultiOutputExactGP/' if (save_plots == 'test' or save_plots == 'all') else None
    )