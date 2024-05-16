import gpytorch
import random
import torch

from GPModels.ExactGPModels import *
from GPModels.MultiOutputExactGPModels import *
from GPModels.SVGPModels import *

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

        # Save the model and Train labels
        torch.save(model.state_dict(),f'models/ExactGP/model-{axis}-{name}.pth')
        torch.save(likelihood.state_dict(),f'models/ExactGP/likelihood-{axis}-{name}.pth')

    # Save the train data and labels 
    torch.save(train_data.clone(),f'data/ExactGP/train_data-{name}.pt')
    torch.save(train_labels[0].clone(),f'data/ExactGP/train_labels-x-{name}.pt')
    torch.save(train_labels[1].clone(),f'data/ExactGP/train_labels-y-{name}.pt')

    print(f'Training of ExactGP model with {name} kernel complete!\n')

def __train_ExactMultiOutputExactGP(train_data, train_labels, model, likelihood, name, training_iter):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

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

def __train_ExactDeepKernel(train_data, x_labels, y_labels, training_iter, device):

    likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model_x = ExactGPModelDeepKernel(train_data, x_labels, likelihood_x).to(device)
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model_y = ExactGPModelDeepKernel(train_data, y_labels, likelihood_y).to(device)

    optimizer_x = torch.optim.Adam([
        {'params': model_x.feature_extractor.parameters()},
        {'params': model_x.covar_module.parameters()},
        {'params': model_x.mean_module.parameters()},
        {'params': model_x.likelihood.parameters()},
    ], lr=0.01)
    optimizer_y = torch.optim.Adam([
        {'params': model_y.feature_extractor.parameters()},
        {'params': model_y.covar_module.parameters()},
        {'params': model_y.mean_module.parameters()},
        {'params': model_y.likelihood.parameters()},
    ], lr=0.01)

    mll_x = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_x, model_x)
    mll_y = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_y, model_y)

    print('Training DeepKernel ExactGP model on {} iterations'.format(training_iter))
    for i in range(training_iter):
        # Zero backprop gradients
        optimizer_x.zero_grad()
        optimizer_y.zero_grad()
        # Get output from model
        output_x = model_x(train_data)
        output_y = model_y(train_data)
        # Compute loss and backprop derivatives
        loss_x = -mll_x(output_x, x_labels)
        loss_y = -mll_y(output_y, y_labels)
        loss_x.backward()
        loss_y.backward()
        optimizer_x.step()
        optimizer_y.step()

        print('|{}{}| {:.2f}% (Iteration {}/{})'.format('█'*int(20*(i+1)/training_iter),' '*(20-int(20*(i+1)/training_iter)),(i+1)*100/training_iter,i+1,training_iter),end='\r')
    print('')
    print(f'Training of DeepeKernel ExactGP complete!\n')

    # Save the model
    torch.save(model_x.state_dict(),f'models/ExactGP/model-x-deep-kernel.pth')
    torch.save(likelihood_x.state_dict(),f'models/ExactGP/likelihood-x-deep-kernel.pth')
    torch.save(model_y.state_dict(),f'models/ExactGP/model-y-deep-kernel.pth')
    torch.save(likelihood_y.state_dict(),f'models/ExactGP/likelihood-y-deep-kernel.pth')

    # Save the train data and labels 
    torch.save(train_data.clone(),f'data/ExactGP/train_data-deep-kernel.pt')
    torch.save(x_labels.clone(),f'data/ExactGP/train_labels-x-deep-kernel.pt')
    torch.save(y_labels.clone(),f'data/ExactGP/train_labels-y-deep-kernel.pt')


def train_ExactGP(gp_data, x_labels, y_labels, options, device, training_iter=10000):
    idxs = torch.IntTensor(random.sample(range(0,len(gp_data)),200))

    # RBF kernel
    if options['RBF']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = ExactGPModelRBF(train_data,train_x_labels,likelihood_x).to(device)
        model_y = ExactGPModelRBF(train_data,train_y_labels,likelihood_y).to(device)
        __train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF',training_iter)

    # RBF + Periodic kernel
    if options['RBF-Periodic']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = ExactGPModelRBFPeriodic(train_data,train_x_labels,likelihood_x).to(device)
        model_y = ExactGPModelRBFPeriodic(train_data,train_y_labels,likelihood_y).to(device)
        __train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF-Periodic',training_iter)

    # Matern 3/2 kernel
    if options['Matern-32']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = ExactGPModelMatern_32(train_data,train_x_labels,likelihood_x).to(device)
        model_y = ExactGPModelMatern_32(train_data,train_y_labels,likelihood_y).to(device)
        __train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-32',training_iter)

    # Matern 5/2 kernel
    if options['Matern-52']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = ExactGPModelMatern_52(train_data,train_x_labels,likelihood_x).to(device)
        model_y = ExactGPModelMatern_52(train_data,train_y_labels,likelihood_y).to(device)
        __train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-52',training_iter)

    # Spectral Mixture (n=3) kernel
    if options['SpectralMixture-3']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = ExactGPModelSpectralMixture_3(train_data,train_x_labels,likelihood_x).to(device)
        model_y = ExactGPModelSpectralMixture_3(train_data,train_y_labels,likelihood_y).to(device)
        __train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-3',training_iter)

    # Spectral Mixture (n=5) kernel
    if options['SpectralMixture-5']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = ExactGPModelSpectralMixture_5(train_data,train_x_labels,likelihood_x).to(device)
        model_y = ExactGPModelSpectralMixture_5(train_data,train_y_labels,likelihood_y).to(device)
        __train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-5',training_iter)

    # Spectral Mixture (n=10) kernel
    if options['SpectralMixture-10']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = ExactGPModelSpectralMixture_10(train_data,train_x_labels,likelihood_x).to(device)
        model_y = ExactGPModelSpectralMixture_10(train_data,train_y_labels,likelihood_y).to(device)
        __train_ExactGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-10',training_iter)

    # DeepKernel
    if options['DeepKernel']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        __train_ExactDeepKernel(train_data,train_x_labels,train_y_labels,training_iter,device)

def train_MultiOutputExactGP(gp_data, x_labels, y_labels, options, device, training_iter=10000):
    '''
    Labels are of the form [x_label, y_label]
    '''
    idxs = torch.IntTensor(random.sample(range(0,len(gp_data)),200))

    # RBF
    if options['RBF']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        ## Stack labels to get single 2-dimensional label
        train_labels = torch.stack([train_x_labels,train_y_labels],dim=1).to(device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(device)
        model = MultiOutputExactGPModelRBF(train_data, train_labels, likelihood).to(device)
        __train_ExactMultiOutputExactGP(train_data,train_labels,model,likelihood,'RBF',training_iter)

    # RBF+Periodic
    if options['RBF-Periodic']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        ## Stack labels to get single 2-dimensional label
        train_labels = torch.stack([train_x_labels,train_y_labels],dim=1).to(device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(device)
        model = MultiOutputExactGPModelRBFPeriodic(train_data, train_labels, likelihood).to(device)
        __train_ExactMultiOutputExactGP(train_data,train_labels,model,likelihood,'RBF-Periodic',training_iter)

    # Matern 3/2
    if options['Matern-32']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        ## Stack labels to get single 2-dimensional label
        train_labels = torch.stack([train_x_labels,train_y_labels],dim=1).to(device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(device)
        model = MultiOutputExactGPModelMatern_32(train_data, train_labels, likelihood).to(device)
        __train_ExactMultiOutputExactGP(train_data,train_labels,model,likelihood,'Matern-32',training_iter)

    # Matern 5/2
    if options['Matern-52']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        ## Stack labels to get single 2-dimensional label
        train_labels = torch.stack([train_x_labels,train_y_labels],dim=1).to(device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(device)
        model = MultiOutputExactGPModelMatern_52(train_data, train_labels, likelihood).to(device)
        __train_ExactMultiOutputExactGP(train_data,train_labels,model,likelihood,'Matern-52',training_iter)

    # Spectral mixture 3
    if options['SpectralMixture-3']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        ## Stack labels to get single 2-dimensional label
        train_labels = torch.stack([train_x_labels,train_y_labels],dim=1).to(device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(device)
        model = MultiOutputExactGPModelSpectralMixture_3(train_data, train_labels, likelihood).to(device)
        __train_ExactMultiOutputExactGP(train_data,train_labels,model,likelihood,'SpectralMixture-3',training_iter)

    # Spectral mixture 5
    if options['SpectralMixture-5']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        ## Stack labels to get single 2-dimensional label
        train_labels = torch.stack([train_x_labels,train_y_labels],dim=1).to(device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(device)
        model = MultiOutputExactGPModelSpectralMixture_5(train_data, train_labels, likelihood).to(device)
        __train_ExactMultiOutputExactGP(train_data,train_labels,model,likelihood,'SpectralMixture-5',training_iter)

    # Spectral mixture 10
    if options['SpectralMixture-10']:
        train_data = torch.FloatTensor(gp_data).index_select(0,idxs).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).index_select(0,idxs).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).index_select(0,idxs).clone().to(device)
        ## Stack labels to get single 2-dimensional label
        train_labels = torch.stack([train_x_labels,train_y_labels],dim=1).to(device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(device)
        model = MultiOutputExactGPModelSpectralMixture_10(train_data, train_labels, likelihood).to(device)
        __train_ExactMultiOutputExactGP(train_data,train_labels,model,likelihood,'SpectralMixture-10',training_iter)

def train_SVGP(gp_data, x_labels, y_labels, options, device, training_iter=10000):
    # RBF kernel
    if options['RBF']:
        train_data = torch.FloatTensor(gp_data).to(device)
        inducing_points = train_data[:200]
        train_x_labels = torch.FloatTensor(x_labels).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = SVGPModelRBF(inducing_points).to(device)
        model_y = SVGPModelRBF(inducing_points).to(device)
        __train_SVGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF',training_iter)

    # RBF + Periodic kernel
    if options['RBF-Periodic']:
        inducing_points = torch.FloatTensor(gp_data[:200]).to(device)
        train_data = torch.FloatTensor(gp_data).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = SVGPModelRBFPeriodic(inducing_points).to(device)
        model_y = SVGPModelRBFPeriodic(inducing_points).to(device)
        __train_SVGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF-Periodic',training_iter)
        torch.save(inducing_points,'data/SVGP/inducing_points-RBF-Periodic.pt')

    # Matern 3/2 kernel
    if options['Matern-32']:
        inducing_points = torch.FloatTensor(gp_data[:200]).to(device)
        train_data = torch.FloatTensor(gp_data).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = SVGPModelMatern_32(inducing_points).to(device)
        model_y = SVGPModelMatern_32(inducing_points).to(device)
        __train_SVGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-32',training_iter)
        torch.save(inducing_points,'data/SVGP/inducing_points-Matern-32.pt')

    # Matern 5/2 kernel
    if options['Matern-52']:
        inducing_points = torch.FloatTensor(gp_data[:200]).to(device)
        train_data = torch.FloatTensor(gp_data).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = SVGPModelMatern_52(inducing_points).to(device)
        model_y = SVGPModelMatern_52(inducing_points).to(device)
        __train_SVGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-52',training_iter)
        torch.save(inducing_points,'data/SVGP/inducing_points-Matern-52.pt')

    # SpectralMixture-3 kernel
    if options['SpectralMixture-3']:
        inducing_points = torch.FloatTensor(gp_data[:200]).to(device)
        train_data = torch.FloatTensor(gp_data).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = SVGPModelSpectralMixture_3(inducing_points).to(device)
        model_y = SVGPModelSpectralMixture_3(inducing_points).to(device)
        __train_SVGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-3',training_iter)
        torch.save(inducing_points,'data/SVGP/inducing_points-SpectralMixture-3.pt')

    # SpectralMixture-5 kernel
    if options['SpectralMixture-5']:
        inducing_points = torch.FloatTensor(gp_data[:200]).to(device)
        train_data = torch.FloatTensor(gp_data).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = SVGPModelSpectralMixture_5(inducing_points).to(device)
        model_y = SVGPModelSpectralMixture_5(inducing_points).to(device)
        __train_SVGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-5',training_iter)
        torch.save(inducing_points,'data/SVGP/inducing_points-SpectralMixture-5.pt')

    # SpectralMixture-10 kernel
    if options['SpectralMixture-10']:
        inducing_points = torch.FloatTensor(gp_data[:200]).to(device)
        train_data = torch.FloatTensor(gp_data).clone().to(device)
        train_x_labels = torch.FloatTensor(x_labels).clone().to(device)
        train_y_labels = torch.FloatTensor(y_labels).clone().to(device)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_x = SVGPModelSpectralMixture_10(inducing_points).to(device)
        model_y = SVGPModelSpectralMixture_10(inducing_points).to(device)
        __train_SVGP(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-10',training_iter)
        torch.save(inducing_points,'data/SVGP/inducing_points-SpectralMixture-10.pt')