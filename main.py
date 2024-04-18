import matplotlib.pyplot as plt
import numpy as np
import json
from Fan import Fan
from System import System
from PID import PID
import argparse
from Trajectory import Trajectory
import os
from pathlib import Path
from ExactGPModel import ExactGPModel
import gpytorch
import torch
import random


# Argument Parser
# TODO check for argument correctness
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
    pass

#---------------------------------------------------------------------------#
#                        Gaussian Process Regression                        #
#---------------------------------------------------------------------------#
