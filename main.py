import argparse
import json
import torch
import warnings
import os
import matplotlib.pyplot as plt

from WindField import WindField
from utils.train import *
from utils.test import *
from utils.exceptions import InvalidCommandLineArgumentException
from modules.Trajectory import Trajectory

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

if show_plots == 'all':
    fig, ax = plt.subplots(2,3)
    fig.set_size_inches(16,9)
    fig.tight_layout(pad=3)
    i = j = 0
    for file in sorted(os.listdir('trajectories')):
        trajectory = Trajectory('trajectories/'+file,1)
        trajectory.plot(Path(file).stem,ax[i//3,j%3])
        i+=1
        j+=1
    plt.show()
    fig.savefig('imgs/wind_fields/trajectories.png',dpi=300)
    plt.close('all')

# Train Models
if not test:
    wind_field = WindField('configs/wind_field.json')

    # Run the simulation for the different trajectories
    for file in os.listdir('trajectories'):
        wind_field.set_trajectory('trajectories/'+file,file)
        wind_field.simulate_wind_field()
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
    test_exact_mogp(
        wind_field,
        'trajectories',
        mo_exact_gp_options,
        window_size = 20,
        laps = 1,
        show = True if (show_plots == 'test' or show_plots == 'all') else False,
        save = 'imgs/' if (save_plots == 'test' or save_plots == 'all') else None
    )