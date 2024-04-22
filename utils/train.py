import gpytorch
import random
import torch

from GPModels.ExactGPModels import *
from GPModels.MultiOutputExactGPModels import *
from GPModels.SVGP import *

def __train_ExactGP(train_data, train_labels, models, likelihoods, name, training_iter):
    '''
    Trains the exact GP model given the specified kernel.\\
    The trained model is saved in the `models/ExactGP` folder
    '''
    axis = ['x','y']
    l = [(models[i], likelihoods[i], train_labels[i], axis[i]) for i in range(2)]

    for model,likelihood,labels,axis in l:
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

        # Save the model
        torch.save(model.state_dict(),f'models/ExactGP/model-{axis}-{name}.pth')
        torch.save(likelihood.state_dict(),f'models/ExactGP/likelihood-{axis}-{name}.pth')

    print(f'Training of ExactGP model with {name} kernel complete!\n')

def __train_ExactMultiOutputExactGP(train_data, train_labels, model, likelihood, name, training_iter):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print('Training ExactGP model on {} iterations using {} kernel'.format(training_iter,name))
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
        ], lr=0.01)
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

    print(f'Training of SVGP model with {name} kernel complete!\n')



def train_ExactGP(gp_data, x_labels, y_labels, training_iter=10000):
    idxs = torch.IntTensor(random.sample(range(0,len(gp_data)),200))

    # RBF kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelRBF(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelRBF(train_data,train_y_labels,likelihood_y)
    __train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF',training_iter)

    # RBF + Periodic kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelRBFPeriodic(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelRBFPeriodic(train_data,train_y_labels,likelihood_y)
    __train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF-Periodic',training_iter)

    # Matern 3/2 kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelMatern_32(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelMatern_32(train_data,train_y_labels,likelihood_y)
    __train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-32',training_iter)

    # Matern 5/2 kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelMatern_52(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelMatern_52(train_data,train_y_labels,likelihood_y)
    __train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-52',training_iter)

    # Spectral Mixture (n=4) kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelSpectralMixture_3(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelSpectralMixture_3(train_data,train_y_labels,likelihood_y)
    __train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-3',training_iter)

def train_MultiOutputExactGP(gp_data, x_labels, y_labels, training_iter=10000):
    '''
    Labels are of the form [x_label, y_label]
    '''
    idxs = torch.IntTensor(random.sample(range(0,len(gp_data)),200))

    # RBF
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    ## Stack labels to get single 2-dimensional label
    train_labels = torch.stack([train_x_labels,train_y_labels],dim=1)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultiOutputExactGPModelRBF(train_data, train_labels, likelihood)
    __train_ExactMultiOutputExactGP(train_data,train_labels,model,likelihood,'RBF',training_iter)

    # RBF+Periodic
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    ## Stack labels to get single 2-dimensional label
    train_labels = torch.stack([train_x_labels,train_y_labels],dim=1)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultiOutputExactGPModelRBFPeriodic(train_data, train_labels, likelihood)
    __train_ExactMultiOutputExactGP(train_data,train_labels,model,likelihood,'RBF-Periodic',training_iter)

    # Matern 3/2
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    ## Stack labels to get single 2-dimensional label
    train_labels = torch.stack([train_x_labels,train_y_labels],dim=1)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultiOutputExactGPModelMatern_32(train_data, train_labels, likelihood)
    __train_ExactMultiOutputExactGP(train_data,train_labels,model,likelihood,'Matern-32',training_iter)

    # Matern 5/2
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    ## Stack labels to get single 2-dimensional label
    train_labels = torch.stack([train_x_labels,train_y_labels],dim=1)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultiOutputExactGPModelMatern_52(train_data, train_labels, likelihood)
    __train_ExactMultiOutputExactGP(train_data,train_labels,model,likelihood,'Matern-52',training_iter)

    # Specrtal mixture 3
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    ## Stack labels to get single 2-dimensional label
    train_labels = torch.stack([train_x_labels,train_y_labels],dim=1)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultiOutputExactGPModelSpectralMixture_3(train_data, train_labels, likelihood)
    __train_ExactMultiOutputExactGP(train_data,train_labels,model,likelihood,'SpectralMixture-3',training_iter)

def train_SVGP(gp_data, x_labels, y_labels, training_iter=10000):
    # RBF kernel
    inducing_points = torch.DoubleTensor(gp_data[:200]).clone()
    train_data = torch.DoubleTensor(gp_data).clone()
    train_x_labels = torch.DoubleTensor(x_labels).clone()
    train_y_labels = torch.DoubleTensor(y_labels).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = SVGPModelRBF(inducing_points)
    model_y = SVGPModelRBF(inducing_points)
    __train_SVGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF',training_iter)

    # RBF+Periodic kernel
    inducing_points = torch.DoubleTensor(gp_data[:200]).clone()
    train_data = torch.DoubleTensor(gp_data).clone()
    train_x_labels = torch.DoubleTensor(x_labels).clone()
    train_y_labels = torch.DoubleTensor(y_labels).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = SVGPModelRBFPeriodic(inducing_points)
    model_y = SVGPModelRBFPeriodic(inducing_points)
    __train_SVGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF-Periodic',training_iter)

    # Matern 3/2 kernel
    inducing_points = torch.DoubleTensor(gp_data[:200]).clone()
    train_data = torch.DoubleTensor(gp_data).clone()
    train_x_labels = torch.DoubleTensor(x_labels).clone()
    train_y_labels = torch.DoubleTensor(y_labels).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = SVGPModelMatern_32(inducing_points)
    model_y = SVGPModelMatern_32(inducing_points)
    __train_SVGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-32',training_iter)

    # Matern 5/2 kernel
    inducing_points = torch.DoubleTensor(gp_data[:200]).clone()
    train_data = torch.DoubleTensor(gp_data).clone()
    train_x_labels = torch.DoubleTensor(x_labels).clone()
    train_y_labels = torch.DoubleTensor(y_labels).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = SVGPModelMatern_52(inducing_points)
    model_y = SVGPModelMatern_52(inducing_points)
    __train_SVGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-52',training_iter)

    # SpectralMixture-3 kernel
    inducing_points = torch.DoubleTensor(gp_data[:200]).clone()
    train_data = torch.DoubleTensor(gp_data).clone()
    train_x_labels = torch.DoubleTensor(x_labels).clone()
    train_y_labels = torch.DoubleTensor(y_labels).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = SVGPModelSpectralMixture_3(inducing_points)
    model_y = SVGPModelSpectralMixture_3(inducing_points)
    __train_SVGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-3',training_iter)