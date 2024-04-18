import gpytorch
import torch
import random
import os
import matplotlib.pyplot as plt
import numpy as np

from ExactGPModel import ExactGPModel

def train_ExactGP(train_data_x, train_data_y, x_labels, y_labels, kernel, name, points=200, training_iter=100000):
    '''
    Trains the exact GP model given the specified kernel.\\
    The trained model is saved in the `models` folder
    '''
    train_data_x = torch.FloatTensor(train_data_x)
    train_data_y = torch.FloatTensor(train_data_y)
    x_labels = torch.FloatTensor(x_labels)
    y_labels = torch.FloatTensor(y_labels)

    # Randomly select a certain number of data points
    x_idxs = random.sample(range(0,len(train_data_x)),points)
    y_idxs = random.sample(range(0,len(train_data_y)),points)
    train_data_x = train_data_x.index_select(0,torch.IntTensor(x_idxs))
    train_data_y = train_data_y.index_select(0,torch.IntTensor(y_idxs))
    x_labels = x_labels.index_select(0,torch.IntTensor(x_idxs))
    y_labels = y_labels.index_select(0,torch.IntTensor(y_idxs))

    # Build the models
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModel(train_data_x, x_labels, likelihood_x)
    model_y = ExactGPModel(train_data_y, y_labels, likelihood_y)

    model_x.train()
    model_y.train()
    likelihood_x.train()
    likelihood_y.train()
    optimizer_x = torch.optim.Adam(model_x.parameters(), lr=0.001)
    optimizer_y = torch.optim.Adam(model_y.parameters(), lr=0.001)
    mll_x = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_x, model_x)
    mll_y = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_y, model_y)

    # Train the models
    for i in range(training_iter):
        # Zero gradients from the previous iteration
        optimizer_x.zero_grad()
        optimizer_y.zero_grad()
        # Output from the model
        output_x = model_x(train_data_x)
        output_y = model_y(train_data_y)
        # Compute the loss and backpropagate the gradients
        loss_x = -mll_x(output_x, x_labels)
        loss_y = -mll_y(output_y, y_labels)
        loss_x.backward()
        loss_y.backward()
        optimizer_x.step()
        optimizer_y.step()
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Training the Exact GP model on {} random points for {} iterations\n".format(points,training_iter))
        print('[{}{}] {:.2f}% ({}/{} iterations)'.format('='*int(20*(i+1)/training_iter),' '*(20-int(20*(i+1)/training_iter)),100*(i+1)/training_iter,i+1,training_iter))
        print('')
    print('Training Complete!')

    # Save the models
    torch.save(model_x.state_dict(), f'models/model-x-{name}-ExactGP.pth')
    torch.save(model_y.state_dict(), f'models/model-y-{name}-ExactGP.pth')
    torch.save(likelihood_x.state_dict(), f'models/likelihood-x-{name}-ExactGP.pth')
    torch.save(likelihood_y.state_dict(), f'models/likelihood-y-{name}-ExactGP.pth')

def test_ExactGP(test_data_x, test_data_y, x_labels, y_labels, 
                 x_model_file, y_model_file, x_likelihood_file, y_likelihood_file,
                 T, name, plot=True, save=False):
    '''
    Test a trained GP model.\\
    If `plot` is set to true, the GP results are shown.\\
    If `save` is set to true, the GP results are saved in the `imgs/gp_plots` folder
    '''
    # Build the models
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModel(test_data_x, x_labels, likelihood_x)
    model_y = ExactGPModel(test_data_y, y_labels, likelihood_y)

    model_x_dict = torch.load(x_model_file)
    model_y_dict = torch.load(y_model_file)
    likelihood_x_dict = torch.load(x_likelihood_file)
    likelihood_y_dict = torch.load(y_likelihood_file)
    model_x.load_state_dict(model_x_dict)
    model_y.load_state_dict(model_y_dict)
    likelihood_x.load_state_dict(likelihood_x_dict)
    likelihood_y.load_state_dict(likelihood_y_dict)

    test_data_x = torch.FloatTensor(test_data_x)
    test_data_y = torch.FloatTensor(test_data_y)
    test_label_x = torch.FloatTensor(test_label_x)
    test_label_y = torch.FloatTensor(test_label_y)

     # Test the model
    model_x.eval()
    model_y.eval()
    likelihood_x.eval()
    likelihood_y.eval()

    with torch.no_grad():
        # pred_x = likelihood_x(model_x(test_data_x))
        # pred_y = likelihood_y(model_y(test_data_y))
        pred_x = model_x(test_data_x)
        pred_y = model_y(test_data_y)

    # Initialize x plot
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    rmse = np.sqrt(1/len(pred_x.mean)*np.linalg.norm(pred_x.mean.numpy()-test_label_x.numpy())**2)

    # Get upper and lower confidence bounds
    lower, upper = pred_x.confidence_region()
    ax.plot(np.NaN, np.NaN, '-', color='none', label='RMSE={0:.2f} N'.format(rmse))
    # Plot training data as black stars
    ax.plot(T,test_label_x,color='orange',label='Real Data')
    # Plot predictive means as blue line
    ax.plot(T,pred_x.mean.numpy(), 'b',label='Mean')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(T, lower.numpy(), upper.numpy(), alpha=0.5, color='c',label='Confidence')
    # Plot training points
    ax.legend()
    ax.set_xlabel(r'$t$ $[s]$')
    ax.set_ylabel(r'$F_{wx}$ $[N]$')
    fig.suptitle(f'GP Wind Estimation (x-axis)')
    if save:
        plt.savefig(f'imgs/test_imgs/ExactGP-{name}-x.png',dpi=300)
    plt.show()

    # Initialize y plot
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    rmse = np.sqrt(1/len(pred_y.mean)*np.linalg.norm(pred_y.mean.numpy()-test_label_y.numpy())**2)

    # Get upper and lower confidence bounds
    lower, upper = pred_y.confidence_region()
    ax.plot(np.NaN, np.NaN, '-', color='none', label='RMSE={0:.2f} N'.format(rmse))
    # Plot training data as black stars
    ax.plot(T,test_label_y, color='orange',label='Real Data')
    # Plot predictive means as blue line
    ax.plot(T,pred_y.mean.numpy(), 'b',label='Mean')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(T, lower.numpy(), upper.numpy(), alpha=0.5, color='c',label='Confidence')
    # Plot training points
    ax.legend()
    ax.set_xlabel(r'$t$ $[s]$')
    ax.set_ylabel(r'$F_{wy}$ $[N]$')
    fig.suptitle(f'GP Wind Estimation (y-axis)')
    if save:
        plt.savefig(f'imgs/test_imgs/ExactGP-{name}-y.png',dpi=300)
    plt.show()
    plt.close()