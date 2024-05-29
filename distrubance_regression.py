from WindField import WindField
import os
from GPModels.SVGPModels import SVGPModelRBF
import torch
import gpytorch

wind_field = WindField('configs/wind_field.json','configs/mass.json')
# Run the simulation for the different trajectories
for file in os.listdir('trajectories'):
    wind_field.set_trajectory('trajectories/'+file,file,laps=3)
    wind_field.simulate_wind_field(False)
    wind_field.reset()

# Get GP data
_, x_labels, y_labels, T = wind_field.get_gp_data()
window_size = 10

# Setup  GP data for training
gp_data_x = torch.FloatTensor([
    [x_labels[t] for t in range(window,window+window_size)] for window in range(0,len(T)-window_size)
])
gp_labels_x = torch.FloatTensor([
    x_labels[t] for t in range(window_size,len(T))
])

gp_data_y = torch.FloatTensor([
    [y_labels[t] for t in range(window,window+window_size)] for window in range(0,len(T)-window_size)
])
gp_labels_y = torch.FloatTensor([
    y_labels[t] for t in range(window_size,len(T))
])


# GP models
model_x = SVGPModelRBF(gp_data_x[:200])
model_y = SVGPModelRBF(gp_data_y[:200])

# Likelihoods
likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
likelihood_y = gpytorch.likelihoods.GaussianLikelihood()

train = True
test = True
training_iter=10000

if train:
    # Train x model
    model_x.train()
    likelihood_x.train()
    optimizer = torch.optim.Adam([
        {'params': model_x.parameters()},
        {'params': likelihood_x.parameters()},
    ], lr=0.001)
    mll=gpytorch.mlls.VariationalELBO(likelihood_x, model_x, num_data=gp_labels_x.size(0))

    # Train each model
    print(len(gp_data_x))
    print('Training SVGP model ({}-axis) on {} iterations using {} kernel'.format('x',training_iter,'RBF'))
    for i in range(training_iter):
        # Zero gradients from the previous iteration
        optimizer.zero_grad()
        # Output from the model
        output = model_x(gp_data_x)
        # Compute the loss and backpropagate the gradients
        # We want to maximize the log-likelihood on the parameters, so
        # we minimize -mll
        loss = -mll(output, gp_labels_x)
        loss.backward()
        optimizer.step()
        print('|{}{}| {:.2f}% (Iteration {}/{})'.format('█'*int(20*(i+1)/training_iter),' '*(20-int(20*(i+1)/training_iter)),(i+1)*100/training_iter,i+1,training_iter),end='\r')
    print('')

    # Save the model
    torch.save(model_x.state_dict(),f'svgp_test_models/model-x-RBF.pth')
    torch.save(likelihood_x.state_dict(),f'svgp_test_models/likelihood-x-RBF.pth')
    # Save Training Inducing Points
    inducing_points = model_x.variational_strategy.inducing_points
    torch.save(inducing_points,f'svgp_test_data/inducing_points_x-RBF.pt')

    print(f'Training of SVGP model with RBF kernel complete!\n')

    # Train y model
    model_y.train()
    likelihood_y.train()
    optimizer = torch.optim.Adam([
        {'params': model_y.parameters()},
        {'params': likelihood_y.parameters()},
    ], lr=0.001)
    mll=gpytorch.mlls.VariationalELBO(likelihood_y, model_y, num_data=gp_labels_y.size(0))

    # Train each model
    print('Training SVGP model ({}-axis) on {} iterations using {} kernel'.format('y',training_iter,'RBF'))
    for i in range(training_iter):
        # Zero gradients from the previous iteration
        optimizer.zero_grad()
        # Output from the model
        output = model_y(gp_data_y)
        # Compute the loss and backpropagate the gradients
        # We want to maximize the log-likelihood on the parameters, so
        # we minimize -mll
        loss = -mll(output, gp_labels_y)
        loss.backward()
        optimizer.step()
        print('|{}{}| {:.2f}% (Iteration {}/{})'.format('█'*int(20*(i+1)/training_iter),' '*(20-int(20*(i+1)/training_iter)),(i+1)*100/training_iter,i+1,training_iter),end='\r')
    print('')

    # Save the model
    torch.save(model_y.state_dict(),f'svgp_test_models/model-y-RBF.pth')
    torch.save(likelihood_y.state_dict(),f'svgp_test_models/likelihood-y-RBF.pth')
    # Save Training Inducing Points
    inducing_points = model_x.variational_strategy.inducing_points
    torch.save(inducing_points,f'svgp_test_data/inducing_points_y-RBF.pt')

    print(f'Training of SVGP model with RBF kernel complete!\n')

if test:
    with torch.no_grad():
        #load the models
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_x_dict = torch.load(f'svgp_test_models/likelihood-x-RBF.pth')
        likelihood_y_dict = torch.load(f'svgp_test_models/likelihood-y-RBF.pth')
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        inducing_points_x = torch.load('svgp_test_data/inducing_points_x-RBF.pt')
        inducing_points_y = torch.load('svgp_test_data/inducing_points_y-RBF.pt')
        model_x = SVGPModelRBF(inducing_points_x)
        model_y = SVGPModelRBF(inducing_points_y)
        model_x_dict = torch.load(f'svgp_test_models/model-x-RBF.pth')
        model_y_dict = torch.load(f'svgp_test_models/model-y-RBF.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)

        # Test the models
        wind_field = WindField('configs/wind_field_test.json','configs/mass.json')
        wind_field.set_trajectory('test_trajectories/lemniscate4.mat','lemniscate4')
        wind_field.simulate_disturbance_regression(model_x,model_y,10,10)