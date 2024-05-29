import gpytorch
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

from GPModels.ExactGPModel import *
from GPModels.MultiOutputExactGPModel import *
from GPModels.SVGPModel import *

@torch.no_grad
def __test_offline_ExactGP(test_data, test_labels, models, likelihoods, name, T, save=False, show=False):
    '''
    Test a trained GP model.\\
    '''
    axis = ['x','y']
    l = [(models[i], likelihoods[i], test_labels[i], axis[i]) for i in range(2)]

    for model,likelihood,labels,axis in l:
        model.eval()
        likelihood.eval()
        observed_pred = model(test_data)
        if axis == 'x':
            fx = observed_pred.mean
        else:
            fy = observed_pred.mean
        lower, upper = observed_pred.confidence_region()
        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.suptitle(f'ExactGP Wind Estimation ({axis}-axis) with {name} kernel')
        ax.plot(T,labels,color='orange',label='True Data')
        ax.plot(T,observed_pred.mean.numpy(),'b-',label="Estimated Data")
        ax.fill_between(T, lower.numpy(), upper.numpy(), alpha=0.5, color='cyan',label='Confidence')
        ax.legend()
        if save:
            plt.savefig(f'imgs/gp_plots/ExactGP/{name}-{axis}.png',dpi=300)
            plt.savefig(f'imgs/gp_plots/ExactGP/{name}-{axis}.svg',dpi=300)
        ax.set_xlabel(r'$t$ $[s]$')
        if axis == 'x':
            ax.set_ylabel(r'$F_{w_x}$ $[N]$')
        else:
            ax.set_ylabel(r'$F_{w_y}$ $[N]$')
        
    f = torch.stack([fx,fy],dim=1).numpy()
    lab = torch.stack([test_labels[0],test_labels[1]],dim=1).numpy()
    rmse = np.sqrt(1/len(f)*np.linalg.norm(f-lab)**2)
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    fig.suptitle(f'Wind Estimation along Trajectory using {name} Kernel')
    ax.plot(np.NaN, np.NaN, '-', color='none')
    ax.set_xlabel(r'$x$ $[m]$')
    ax.set_ylabel(r'$y$ $[m]$')
    for i,p in enumerate(test_data):
        if i%10 == 0:
            x = p[0]
            y = p[1]
            ax.arrow(x,y,test_labels[0][i],test_labels[1][i],length_includes_head=False,head_width=0.01,head_length=0.01,width=0.004,color='orange')
            ax.arrow(x,y,fx[i],fy[i],length_includes_head=False,head_width=0.01,head_length=0.01,width=0.004,color='cyan',alpha=0.5)
    fig.legend(['RMSE = {:.2f} N'.format(rmse),'Real Wind Force','Estimated Wind Force'])
    if save:
        plt.savefig(f'imgs/gp_plots/ExactGP/{name}-full-trajectory.png',dpi=300)
        plt.savefig(f'imgs/gp_plots/ExactGP/{name}-full-trajectory.svg',dpi=300)

    if show:
        plt.show()
    plt.close()

@torch.no_grad
def __test_offline_ExactMultiOutputExactGP(test_data, test_labels, model, likelihood, name, T, save=False, show=False):
    model.eval()
    likelihood.eval()
    observed_pred = model(test_data)
    lower, upper = observed_pred.confidence_region()
    
    # x-axis plot
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    fig.suptitle(f'MultiOutputExactGP Wind Estimation (x-axis) with {name} kernel')
    ax.plot(T,test_labels[0],color='orange',label='True Data')
    ax.plot(T,observed_pred.mean[:,0],'b-',label="Estimated Data")
    ax.fill_between(T, lower[:,0].numpy(), upper[:,0].numpy(), alpha=0.5, color='cyan',label='Confidence')
    ax.legend()
    if save:
        plt.savefig(f'imgs/gp_plots/MultiOutputExactGP/{name}-x',dpi=300)

    # y-axis plot
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    fig.suptitle(f'MultiOutputExactGP Wind Estimation (y-axis) with {name} kernel')
    ax.plot(T,test_labels[1],color='orange',label='True Data')
    ax.plot(T,observed_pred.mean[:,1],'b-',label="Estimated Data")
    ax.fill_between(T, lower[:,1].numpy(), upper[:,1].numpy(), alpha=0.5, color='cyan',label='Confidence')
    ax.legend()
    if save:
        plt.savefig(f'imgs/gp_plots/MultiOutputExactGP/{name}-y',dpi=300)

    # Full-trajectory plot
    lab = torch.stack([test_labels[0],test_labels[1]],dim=1).numpy()
    rmse = np.sqrt(1/len(observed_pred.mean)*np.linalg.norm(observed_pred.mean-lab)**2)
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    fig.suptitle(f'Wind Estimation along Trajectory using {name} Kernel')
    ax.plot(np.NaN, np.NaN, '-', color='none')
    ax.set_xlabel(r'$x$ $[m]$')
    ax.set_ylabel(r'$y$ $[m]$')
    for i,p in enumerate(test_data):
        if i%10 == 0:
            x = p[0]
            y = p[1]
            ax.arrow(x,y,test_labels[0][i],test_labels[1][i],length_includes_head=False,head_width=0.01,head_length=0.01,width=0.004,color='orange')
            ax.arrow(x,y,observed_pred.mean[i][0],observed_pred.mean[i][0],length_includes_head=False,head_width=0.01,head_length=0.01,width=0.004,color='cyan',alpha=0.5)
    fig.legend(['RMSE = {:.2f} N'.format(rmse),'Real Wind Force','Estimated Wind Force'])
    if save:
        plt.savefig(f'imgs/gp_plots/MultiOutputExactGP/{name}-full-trajectory.png',dpi=300)
        plt.savefig(f'imgs/gp_plots/MultiOutputExactGP/{name}-full-trajectory.svg',dpi=300)

    if show:
        plt.show()
    plt.close()

@torch.no_grad
def __test_offline_SVGP(test_data, test_labels, models, likelihoods, name, T, trajectory_name, save=False, show=False):
    axis = ['x','y']
    l = [(models[i], likelihoods[i], test_labels[i], axis[i]) for i in range(2)]

    for model,likelihood,labels,axis in l:
        model.eval()
        likelihood.eval()
        observed_pred = model(test_data)
        if axis == 'x':
            fx = observed_pred.mean
        else:
            fy = observed_pred.mean
        lower, upper = observed_pred.confidence_region()
        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.suptitle(f'SVGP Wind Estimation ({axis}-axis) with {name} kernel along {trajectory_name} Trajectory')
        ax.plot(T,labels,color='orange',label='True Data')
        ax.plot(T,observed_pred.mean.numpy(),'b-',label="Estimated Data")
        ax.fill_between(T, lower.numpy(), upper.numpy(), alpha=0.5, color='cyan',label='Confidence')
        ax.legend()
        if save:
            plt.savefig(f'imgs/gp_plots/SVGP/{name}-{axis}-{trajectory_name}-trajectory.png',dpi=300)
            plt.savefig(f'imgs/gp_plots/SVGP/{name}-{axis}-{trajectory_name}-trajectory.svg',dpi=300)
        ax.set_xlabel(r'$t$ $[s]$')
        if axis == 'x':
            ax.set_ylabel(r'$F_{w_x}$ $[N]$')
        else:
            ax.set_ylabel(r'$F_{w_y}$ $[N]$')

    f = torch.stack([fx,fy],dim=1).numpy()
    lab = torch.stack([test_labels[0],test_labels[1]],dim=1).numpy()
    rmse = np.sqrt(1/len(f)*np.linalg.norm(f-lab)**2)
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    fig.suptitle(f'Wind Estimation along {trajectory_name} Trajectory using {name} Kernel')
    ax.plot(np.NaN, np.NaN, '-', color='none')
    ax.set_xlabel(r'$x$ $[m]$')
    ax.set_ylabel(r'$y$ $[m]$')
    for i,p in enumerate(test_data):
        if i%10 == 0:
            x = p[0]
            y = p[1]
            ax.arrow(x,y,test_labels[0][i]/5,test_labels[1][i]/5,length_includes_head=False,head_width=0.01,head_length=0.01,width=0.004,color='orange')
            ax.arrow(x,y,fx[i]/5,fy[i]/5,length_includes_head=False,head_width=0.01,head_length=0.01,width=0.004,color='cyan',alpha=0.5)
    fig.legend(['RMSE = {:.2f} N'.format(rmse),'Real Wind Force','Estimated Wind Force'])
    if save:
        plt.savefig(f'imgs/gp_plots/SVGP/{name}-{trajectory_name}-trajectory.png',dpi=300)
        plt.savefig(f'imgs/gp_plots/SVGP/{name}-{trajectory_name}-trajectory.svg',dpi=300)

    # Plot Kernel Heatmap
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(16,9)
    fig.suptitle(f'Kernel Heatmap Using the {name} Kernel')
    # Obtain the Gram matrix associated to the kernel evaluated in the inducing points
    gram_x = models[0].covar_module(models[0].variational_strategy.inducing_points).evaluate()
    gram_y = models[1].covar_module(models[1].variational_strategy.inducing_points).evaluate()
    cb1 = ax[0].imshow(gram_x, cmap='magma', interpolation='nearest')
    cb2 = ax[1].imshow(gram_y, cmap='magma', interpolation='nearest')
    fig.colorbar(cb1,fraction=0.046, pad=0.04, label='Correrlation')
    fig.colorbar(cb2,fraction=0.046, pad=0.04, label='Correrlation')
    ax[0].title.set_text(r'Kernel Heatmap $x$-axis')
    ax[1].title.set_text(r'Kernel Heatmap $y$-axis')
    plt.savefig(f'imgs/gp_plots/SVGP/{name}-heatmap.png',dpi=300)
    plt.savefig(f'imgs/gp_plots/SVGP/{name}-heatmap.svg',dpi=300)

    if show:
        plt.show()
    plt.close()


def test_offline_ExactGP(gp_data,x_labels,y_labels,T,save_plots,options):
    file = open(".metadata/exact_gp_dict","rb")
    exact_gp_dict = pickle.load(file)

    for name in exact_gp_dict:
        if options[name]:
            test_data = torch.FloatTensor(gp_data)
            test_x_labels = torch.FloatTensor(x_labels)
            test_y_labels = torch.FloatTensor(y_labels)
            likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
            likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
            likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-{name}.pth')
            likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-{name}.pth')
            likelihood_x.load_state_dict(likelihood_x_dict)
            likelihood_y.load_state_dict(likelihood_y_dict)
            train_data = torch.load(f'data/ExactGP/train_data-{name}.pt')
            train_labels_x = torch.load(f'data/ExactGP/train_labels-x-{name}.pt')
            train_labels_y = torch.load(f'data/ExactGP/train_labels-y-{name}.pt')
            model_x = exact_gp_dict[name](train_data,train_labels_x,likelihood_x)
            model_y = exact_gp_dict[name](train_data,train_labels_y,likelihood_y)
            model_x_dict = torch.load(f'models/ExactGP/model-x-{name}.pth')
            model_y_dict = torch.load(f'models/ExactGP/model-y-{name}.pth')
            model_x.load_state_dict(model_x_dict)
            model_y.load_state_dict(model_y_dict)
            __test_offline_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],name,T,save=save_plots)

def test_offline_MultiOutputExactGP(gp_data,x_labels,y_labels,T,save_plots,options):
    file = open(".metadata/mo_exact_gp_dict","rb")
    mo_exact_gp_dict = pickle.load(file)

    for name in mo_exact_gp_dict:
        if options[name]:
            test_data = torch.FloatTensor(gp_data)
            test_x_labels = torch.FloatTensor(x_labels)
            test_y_labels = torch.FloatTensor(y_labels)
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
            likelihood_dict = torch.load(f'models/MultiOutputExactGP/likelihood-{name}.pth')
            likelihood.load_state_dict(likelihood_dict)
            train_data = torch.load(f'data/MultiOutputExactGP/train_data-{name}.pt')
            train_labels = torch.load(f'data/MultiOutputExactGP/train_labels-{name}.pt')
            model = mo_exact_gp_dict[name](train_data,train_labels,likelihood)
            model_dict = torch.load(f'models/MultiOutputExactGP/model-{name}.pth')
            model.load_state_dict(model_dict)
            __test_offline_ExactMultiOutputExactGP(test_data,[test_x_labels,test_y_labels],model,likelihood,{name},T,save=save_plots)

def test_offline_SVGP(gp_data,x_labels,y_labels,T,save_plots,options,trajectory_name):
    file = open(".metadata/svgp_dict","rb")
    svgp_dict = pickle.load(file)

    for name in svgp_dict:
        if options[name]:
            test_data = torch.FloatTensor(gp_data)
            test_x_labels = torch.FloatTensor(x_labels)
            test_y_labels = torch.FloatTensor(y_labels)
            likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
            likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
            likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-{name}.pth')
            likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-{name}.pth')
            likelihood_x.load_state_dict(likelihood_x_dict)
            likelihood_y.load_state_dict(likelihood_y_dict)
            inducing_points_x = torch.load(f'data/SVGP/inducing_points_x-{name}.pt')
            inducing_points_y = torch.load(f'data/SVGP/inducing_points_y-{name}.pt')
            model_x = svgp_dict[name](inducing_points_x)
            model_y = svgp_dict[name](inducing_points_y)
            model_x_dict = torch.load(f'models/SVGP/model-x-{name}.pth')
            model_y_dict = torch.load(f'models/SVGP/model-y-{name}.pth')
            model_x.load_state_dict(model_x_dict)
            model_y.load_state_dict(model_y_dict)
            __test_offline_SVGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],name,T,trajectory_name,save=save_plots)