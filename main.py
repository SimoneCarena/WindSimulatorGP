import argparse
import os

from WindField import WindField
from utils.train import *
from utils.test import *

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

wind_field.plot(True)
exit()

# Get GP data
gp_data, x_labels, y_labels, T = wind_field.get_gp_data()

#-----------------------------------------------#
#                Train GP models                #
#-----------------------------------------------#

if not test:
    # train_ExactGP(gp_data,x_labels,y_labels)
    train_MultiOutputExactGP(gp_data,x_labels,y_labels)


#----------------------------------------------#
#                Test GP models                #
#----------------------------------------------#

# test_ExactGP(gp_data,x_labels,y_labels,T,save_plots)
test_MultiOutputExactGP(gp_data,x_labels,y_labels,T,save_plots)