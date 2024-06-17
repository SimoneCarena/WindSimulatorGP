import gpytorch
import random
import torch
import pickle

from GPModels.ExactGPModel import *
from GPModels.MultiOutputExactGPModel import *
from GPModels.SVGPModel import *

def __train_ExactGP(train_data, train_labels, models, likelihoods, name, training_iter):
    '''
    Trains the exact GP model given the specified kernel.\\
    The trained model is saved in the `models/ExactGP` folder
    '''
    axis = ['x','y']
    l = [(models[i], likelihoods[i], train_labels[i], axis[i]) for i in range(2)]

    for model,likelihood,labels,axis in l:
        # for param_name, param in model.named_parameters():
        #     print(f'Parameter name: {param_name:42} value = {param}')
        # Setup each model
        model.train()
        likelihood.train()
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01)
        mll=gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Train each model
        print('Training ExactGP model ({}-axis) on {} iterations using {} kernel'.format(axis,training_iter,name))
        for i in range(training_iter):
            # Zero gradients from the previous iteration
            optimizer.zero_grad()
            # Output from the model
            output = model(train_data)
            # Compute the loss and backpropagate the gradients
            # We want to maximize the log-likelihood on the parameters, so
            # we minimize -mll
            loss = -mll(output, labels)
            loss.backward()
            optimizer.step()
            print('|{}{}| {:.2f}% (Iteration {}/{})'.format('█'*int(20*(i+1)/training_iter),' '*(20-int(20*(i+1)/training_iter)),(i+1)*100/training_iter,i+1,training_iter),end='\r')
        print('')

        # Save the model and Train labels
        torch.save(model.state_dict(),f'models/ExactGP/model-{axis}-{name}.pth')
        torch.save(likelihood.state_dict(),f'models/ExactGP/likelihood-{axis}-{name}.pth')
        # for param_name, param in model.named_parameters():
        #     print(f'Parameter name: {param_name:42} value = {param.item()}')
        # print('')

    # Save the train data and labels 
    torch.save(train_data.clone(),f'data/ExactGP/train_data-{name}.pt')
    torch.save(train_labels[0].clone(),f'data/ExactGP/train_labels-x-{name}.pt')
    torch.save(train_labels[1].clone(),f'data/ExactGP/train_labels-y-{name}.pt')

    print(f'Training of ExactGP model with {name} kernel complete!\n')

def __train_ExactMultiOutputExactGP(train_data, train_labels, model, likelihood, name, training_iter):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print('Training MultiOutputExactGP model on {} iterations using {} kernel'.format(training_iter,name))
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_data)
        loss = -mll(output, train_labels)
        loss.backward()
        optimizer.step()
        print('|{}{}| {:.2f}% (Iteration {}/{})'.format('█'*int(20*(i+1)/training_iter),' '*(20-int(20*(i+1)/training_iter)),(i+1)*100/training_iter,i+1,training_iter),end='\r')
    print('')
    print(f'Training of MultiOutputExactGP model with {name} kernel complete!\n')

    # Save the model
    torch.save(model.state_dict(),f'models/MultiOutputExactGP/model-{name}.pth')
    torch.save(likelihood.state_dict(),f'models/MultiOutputExactGP/likelihood-{name}.pth')
    # Save the train data and labels 
    torch.save(train_data.clone(),f'data/MultiOutputExactGP/train_data-{name}.pt')
    torch.save(train_labels.clone(),f'data/MultiOutputExactGP/train_labels-{name}.pt')

def __train_SVGP(train_data, train_labels, models, likelihoods, name, training_iter):
    axis = ['x','y']
    l = [(models[i], likelihoods[i], train_labels[i], axis[i]) for i in range(2)]

    for model,likelihood,labels,axis in l:
        # Setup each model
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.001)
        mll=gpytorch.mlls.VariationalELBO(likelihood, model, num_data=labels.size(0))

        # Train each model
        print('Training SVGP model ({}-axis) on {} iterations using {} kernel'.format(axis,training_iter,name))
        for i in range(training_iter):
            # Zero gradients from the previous iteration
            optimizer.zero_grad()
            # Output from the model
            output = model(train_data)
            # Compute the loss and backpropagate the gradients
            # We want to maximize the log-likelihood on the parameters, so
            # we minimize -mll
            loss = -mll(output, labels)
            loss.backward()
            optimizer.step()
            print('|{}{}| {:.2f}% (Iteration {}/{})'.format('█'*int(20*(i+1)/training_iter),' '*(20-int(20*(i+1)/training_iter)),(i+1)*100/training_iter,i+1,training_iter),end='\r')
        print('')

        # Save the model
        torch.save(model.state_dict(),f'models/SVGP/model-{axis}-{name}.pth')
        torch.save(likelihood.state_dict(),f'models/SVGP/likelihood-{axis}-{name}.pth')
        # Save Training Inducing Points
        inducing_points = model.variational_strategy.inducing_points
        torch.save(inducing_points,f'data/SVGP/inducing_points_{axis}-{name}.pt')

    print(f'Training of SVGP model with {name} kernel complete!\n')

def train_ExactGP(gp_data, x_labels, y_labels, options, device, training_iter=10000):
    idxs = torch.arange(0,1000)
    file = open(".metadata/exact_gp_dict","rb")
    exact_gp_dict = pickle.load(file)
    
    for name in exact_gp_dict:
        if options[name]:
            train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
            train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
            train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
            likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
            likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model_x = exact_gp_dict[name](train_data,train_x_labels,likelihood_x).to(device)
            model_y = exact_gp_dict[name](train_data,train_y_labels,likelihood_y).to(device)
            __train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],name,training_iter)

def train_MultiOutputExactGP(gp_data, x_labels, y_labels, options, device, training_iter=10000):
    idxs = torch.IntTensor(random.sample(range(len(gp_data)),1000))
    file = open(".metadata/mo_exact_gp_dict","rb")
    mo_exact_gp_dict = pickle.load(file)
    
    for name in mo_exact_gp_dict:
        if options[name]:
            train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
            train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
            train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
            ## Stack labels to get single 2-dimensional label
            train_labels = torch.stack([train_x_labels,train_y_labels],dim=1).to(device)
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(device)
            model = mo_exact_gp_dict[name](train_data, train_labels, likelihood).to(device)
            __train_ExactMultiOutputExactGP(train_data,train_labels,model,likelihood,name,training_iter)

def train_SVGP(gp_data, x_labels, y_labels, options, device, training_iter=10000):
    file = open(".metadata/svgp_dict","rb")
    svgp_dict = pickle.load(file)
    
    for name in svgp_dict:
        if options[name]:
            train_data = torch.FloatTensor(gp_data).to(device)
            inducing_points = train_data[:200]
            train_x_labels = torch.FloatTensor(x_labels).clone().to(device)
            train_y_labels = torch.FloatTensor(y_labels).clone().to(device)
            likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
            likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model_x = svgp_dict[name](inducing_points).to(device)
            model_y = svgp_dict[name](inducing_points).to(device)
            __train_SVGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],name,training_iter)