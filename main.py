import numpy as np
import argparse
import os

from WindField import WindField
from gp_utils import *
from GPModels.ExactGPModel import ExactGPModel
from GPModels.MultiOutputExactGPModel import MultiOutputExactGPModel

# Argument Parser
# TODO check for argument correctness and maybe add Exceptions
parser = argparse.ArgumentParser()
parser.add_argument('--save_plots',action='store_true')
parser.add_argument('--show_plots',action='store_true')
parser.add_argument('--test','-t',action='store_true')

# Parse Arguments
args = parser.parse_args()
save_plots = args.save_plots
show_plots = args.show_plots
test = args.test

# Run the simulation for the different trajectories
for file in os.listdir('trajectories'):
    wind_field = WindField('configs/wind_field.json','configs/mass.json','trajectories/'+file)

gp_data, x_labels, y_labels = wind_field.get_gp_data()

#---------------------------------------------------------------------------#
#                        Gaussian Process Regression                        #
#---------------------------------------------------------------------------#

