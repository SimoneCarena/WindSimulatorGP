import gpytorch
import torch
import matplotlib.pyplot as plt
import numpy as np

from GPModels.ExactGPModels import *
from GPModels.MultiOutputExactGPModels import *
from GPModels.SVGPModels import *

@torch.no_grad
def __test_ExactGP(test_data, test_labels, models, likelihoods, name, T, save=False, show=True):
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
    fig.suptitle(f'Wind Estimation along Square Trajectory using {name} Kernel')
    ax.plot(np.NaN, np.NaN, '-', color='none')
    ax.set_xlabel(r'$x$ $[m]$')
    ax.set_ylabel(r'$y$ $[m]$')
    for i,p in enumerate(test_data):
        if i%10 == 0:
            x = p[0]
            y = p[1]
            ax.arrow(x,y,test_labels[0][i],test_labels[1][i],length_includes_head=False,head_width=0.01,head_length=0.01,width=0.004,color='orange')
            ax.arrow(x,y,fx[i],fy[i],length_includes_head=False,head_width=0.01,head_length=0.01,width=0.004,color='b',alpha=0.5)
    fig.legend(['RMSE = {:.2f} N'.format(rmse),'Real Wind Force','Estimated Wind Force'])
    if save:
        plt.savefig(f'imgs/gp_plots/ExactGP/{name}-full-trajectory.png',dpi=300)
        plt.savefig(f'imgs/gp_plots/ExactGP/{name}-full-trajectory.svg',dpi=300)

    if show:
        plt.show()
    plt.close()

@torch.no_grad
def __test_ExactMultiOutputExactGP(test_data, test_labels, model, likelihood, name, T, save=False, show=True):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
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

    if show:
        plt.show()
    plt.close()

@torch.no_grad
def __test_SVGP(test_data, test_labels, models, likelihoods, name, T, save=False, show=True):
    axis = ['x','y']
    l = [(models[i], likelihoods[i], test_labels[i], axis[i]) for i in range(2)]

    for model,likelihood,labels,axis in l:
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            observed_pred = model(test_data)
        lower, upper = observed_pred.confidence_region()
        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.suptitle(f'SVGP Wind Estimation ({axis}-axis) with {name} kernel')
        ax.plot(T,labels,color='orange',label='True Data')
        ax.plot(T,observed_pred.mean.numpy(),'b-',label="Estimated Data")
        ax.fill_between(T, lower.numpy(), upper.numpy(), alpha=0.5, color='cyan',label='Confidence')
        ax.legend()
        if save:
            plt.savefig(f'imgs/gp_plots/SVGP/{name}-{axis}',dpi=300)

    if show:
        plt.show()
    plt.close()


def test_ExactGP(gp_data,x_labels,y_labels,T,save_plots,options):
    # RBF
    if options['RBF'] == True:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = ExactGPModelRBF(test_data,test_x_labels,likelihood_x)
        model_y = ExactGPModelRBF(test_data,test_y_labels,likelihood_y)
        model_x_dict = torch.load(f'models/ExactGP/model-x-RBF.pth')
        model_y_dict = torch.load(f'models/ExactGP/model-y-RBF.pth')
        likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-RBF.pth')
        likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-RBF.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF',T,save=save_plots)

    # RBF + Periodic
    if options['RBF-Periodic']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = ExactGPModelRBFPeriodic(test_data,test_x_labels,likelihood_x)
        model_y = ExactGPModelRBFPeriodic(test_data,test_y_labels,likelihood_y)
        model_x_dict = torch.load(f'models/ExactGP/model-x-RBF-Periodic.pth')
        model_y_dict = torch.load(f'models/ExactGP/model-y-RBF-Periodic.pth')
        likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-RBF-Periodic.pth')
        likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-RBF-Periodic.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF-Periodic',T,save=save_plots)

    # Matern 3/2
    if options['Matern-32']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = ExactGPModelMatern_32(test_data,test_x_labels,likelihood_x)
        model_y = ExactGPModelMatern_32(test_data,test_y_labels,likelihood_y)
        model_x_dict = torch.load(f'models/ExactGP/model-x-Matern-32.pth')
        model_y_dict = torch.load(f'models/ExactGP/model-y-Matern-32.pth')
        likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-Matern-32.pth')
        likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-Matern-32.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-32',T,save=save_plots)

    # Matern 5/2
    if options['Matern-52']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = ExactGPModelMatern_52(test_data,test_x_labels,likelihood_x)
        model_y = ExactGPModelMatern_52(test_data,test_y_labels,likelihood_y)
        model_x_dict = torch.load(f'models/ExactGP/model-x-Matern-52.pth')
        model_y_dict = torch.load(f'models/ExactGP/model-y-Matern-52.pth')
        likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-Matern-52.pth')
        likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-Matern-52.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-52',T,save=save_plots)

    # Spectral Mixture (n=3)
    if options['SpectralMixture-3']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = ExactGPModelSpectralMixture_3(test_data,test_x_labels,likelihood_x)
        model_y = ExactGPModelSpectralMixture_3(test_data,test_y_labels,likelihood_y)
        model_x_dict = torch.load(f'models/ExactGP/model-x-SpectralMixture-3.pth')
        model_y_dict = torch.load(f'models/ExactGP/model-y-SpectralMixture-3.pth')
        likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-SpectralMixture-3.pth')
        likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-SpectralMixture-3.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-3',T,save=save_plots)\
        
    # Spectral Mixture (n=5)
    if options['SpectralMixture-5']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = ExactGPModelSpectralMixture_5(test_data,test_x_labels,likelihood_x)
        model_y = ExactGPModelSpectralMixture_5(test_data,test_y_labels,likelihood_y)
        model_x_dict = torch.load(f'models/ExactGP/model-x-SpectralMixture-5.pth')
        model_y_dict = torch.load(f'models/ExactGP/model-y-SpectralMixture-5.pth')
        likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-SpectralMixture-5.pth')
        likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-SpectralMixture-5.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-5',T,save=save_plots)

    # Spectral Mixture (n=10)
    if options['SpectralMixture-10']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = ExactGPModelSpectralMixture_10(test_data,test_x_labels,likelihood_x)
        model_y = ExactGPModelSpectralMixture_10(test_data,test_y_labels,likelihood_y)
        model_x_dict = torch.load(f'models/ExactGP/model-x-SpectralMixture-10.pth')
        model_y_dict = torch.load(f'models/ExactGP/model-y-SpectralMixture-10.pth')
        likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-SpectralMixture-10.pth')
        likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-SpectralMixture-10.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-10',T,save=save_plots)

    # DeepKernel
    if options['DeepKernel']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = ExactGPModelDeepKernel(test_data,test_x_labels,likelihood_x)
        model_y = ExactGPModelDeepKernel(test_data,test_y_labels,likelihood_y)
        model_x_dict = torch.load(f'models/ExactGP/model-x-deep-kernel.pth')
        model_y_dict = torch.load(f'models/ExactGP/model-y-deep-kernel.pth')
        likelihood_x_dict = torch.load(f'models/ExactGP/likelihood-x-deep-kernel.pth')
        likelihood_y_dict = torch.load(f'models/ExactGP/likelihood-y-deep-kernel.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_ExactGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'DeepKernel',T,save=save_plots)

def test_MultiOutputExactGP(gp_data,x_labels,y_labels,T,save_plots,options):
    # RBF
    if options['RBF']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        test_labels = torch.stack([test_x_labels,test_y_labels],dim=1)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultiOutputExactGPModelRBF(test_data,test_labels,likelihood)
        model_dict = torch.load(f'models/MultiOutputExactGP/model-RBF.pth')
        likelihood_dict = torch.load(f'models/MultiOutputExactGP/likelihood-RBF.pth')
        model.load_state_dict(model_dict)
        likelihood.load_state_dict(likelihood_dict)
        __test_ExactMultiOutputExactGP(test_data,[test_x_labels,test_y_labels],model,likelihood,'RBF',T,save=save_plots)

    # RBF+Periodic
    if options['RBF-Periodic']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        test_labels = torch.stack([test_x_labels,test_y_labels],dim=1)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultiOutputExactGPModelRBFPeriodic(test_data,test_labels,likelihood)
        model_dict = torch.load(f'models/MultiOutputExactGP/model-RBF-Periodic.pth')
        likelihood_dict = torch.load(f'models/MultiOutputExactGP/likelihood-RBF-Periodic.pth')
        model.load_state_dict(model_dict)
        likelihood.load_state_dict(likelihood_dict)
        __test_ExactMultiOutputExactGP(test_data,[test_x_labels,test_y_labels],model,likelihood,'RBF-Periodic',T,save=save_plots)

    # Matern 3/2
    if options['Matern-32']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        test_labels = torch.stack([test_x_labels,test_y_labels],dim=1)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultiOutputExactGPModelMatern_32(test_data,test_labels,likelihood)
        model_dict = torch.load(f'models/MultiOutputExactGP/model-Matern-32.pth')
        likelihood_dict = torch.load(f'models/MultiOutputExactGP/likelihood-Matern-32.pth')
        model.load_state_dict(model_dict)
        likelihood.load_state_dict(likelihood_dict)
        __test_ExactMultiOutputExactGP(test_data,[test_x_labels,test_y_labels],model,likelihood,'Matern-32',T,save=save_plots)

    # Matern 5/2
    if options['Matern-52']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        test_labels = torch.stack([test_x_labels,test_y_labels],dim=1)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultiOutputExactGPModelMatern_52(test_data,test_labels,likelihood)
        model_dict = torch.load(f'models/MultiOutputExactGP/model-Matern-52.pth')
        likelihood_dict = torch.load(f'models/MultiOutputExactGP/likelihood-Matern-52.pth')
        model.load_state_dict(model_dict)
        likelihood.load_state_dict(likelihood_dict)
        __test_ExactMultiOutputExactGP(test_data,[test_x_labels,test_y_labels],model,likelihood,'Matern-52',T,save=save_plots)

    # Spectral Mixture 3
    if options['SpectralMixture-3']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        test_labels = torch.stack([test_x_labels,test_y_labels],dim=1)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultiOutputExactGPModelSpectralMixture_3(test_data,test_labels,likelihood)
        model_dict = torch.load(f'models/MultiOutputExactGP/model-SpectralMixture-3.pth')
        likelihood_dict = torch.load(f'models/MultiOutputExactGP/likelihood-SpectralMixture-3.pth')
        model.load_state_dict(model_dict)
        likelihood.load_state_dict(likelihood_dict)
        __test_ExactMultiOutputExactGP(test_data,[test_x_labels,test_y_labels],model,likelihood,'SpectralMixture-3',T,save=save_plots)

    # Spectral Mixture 5
    if options['SpectralMixture-5']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        test_labels = torch.stack([test_x_labels,test_y_labels],dim=1)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultiOutputExactGPModelSpectralMixture_5(test_data,test_labels,likelihood)
        model_dict = torch.load(f'models/MultiOutputExactGP/model-SpectralMixture-5.pth')
        likelihood_dict = torch.load(f'models/MultiOutputExactGP/likelihood-SpectralMixture-5.pth')
        model.load_state_dict(model_dict)
        likelihood.load_state_dict(likelihood_dict)
        __test_ExactMultiOutputExactGP(test_data,[test_x_labels,test_y_labels],model,likelihood,'SpectralMixture-5',T,save=save_plots)

    # Spectral Mixture 10
    if options['SpectralMixture-10']:
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        test_labels = torch.stack([test_x_labels,test_y_labels],dim=1)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultiOutputExactGPModelSpectralMixture_10(test_data,test_labels,likelihood)
        model_dict = torch.load(f'models/MultiOutputExactGP/model-SpectralMixture-10.pth')
        likelihood_dict = torch.load(f'models/MultiOutputExactGP/likelihood-SpectralMixture-10.pth')
        model.load_state_dict(model_dict)
        likelihood.load_state_dict(likelihood_dict)
        __test_ExactMultiOutputExactGP(test_data,[test_x_labels,test_y_labels],model,likelihood,'SpectralMixture-10',T,save=save_plots)

def test_SVGP(gp_data,x_labels,y_labels,T,save_plots,options):
    # RBF
    if options['RBF']:
        inducing_points = torch.FloatTensor(gp_data[:200]).clone()
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = SVGPModelRBF(inducing_points)
        model_y = SVGPModelRBF(inducing_points)
        model_x_dict = torch.load(f'models/SVGP/model-x-RBF.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-RBF.pth')
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-RBF.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-RBF.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_SVGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF',T,save=save_plots)

    # RBF+Periodic
    if options['RBF-Periodic']:
        inducing_points = torch.FloatTensor(gp_data[:200]).clone()
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = SVGPModelRBFPeriodic(inducing_points)
        model_y = SVGPModelRBFPeriodic(inducing_points)
        model_x_dict = torch.load(f'models/SVGP/model-x-RBF-Periodic.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-RBF-Periodic.pth')
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-RBF-Periodic.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-RBF-Periodic.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_SVGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF-Periodic',T,save=save_plots)

    # Matern 3/2
    if options['Matern-32']:
        inducing_points = torch.FloatTensor(gp_data[:200]).clone()
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = SVGPModelMatern_32(inducing_points)
        model_y = SVGPModelMatern_32(inducing_points)
        model_x_dict = torch.load(f'models/SVGP/model-x-Matern-32.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-Matern-32.pth')
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-Matern-32.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-Matern-32.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_SVGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-32',T,save=save_plots)

    # Matern 5/2
    if options['Matern-52']:
        inducing_points = torch.FloatTensor(gp_data[:200]).clone()
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = SVGPModelMatern_52(inducing_points)
        model_y = SVGPModelMatern_52(inducing_points)
        model_x_dict = torch.load(f'models/SVGP/model-x-Matern-52.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-Matern-52.pth')
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-Matern-52.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-Matern-52.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_SVGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-52',T,save=save_plots)

    # SpectralMixture-3
    if options['SpectralMixture-3']:
        inducing_points = torch.FloatTensor(gp_data[:200]).clone()
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = SVGPModelSpectralMixture_3(inducing_points)
        model_y = SVGPModelSpectralMixture_3(inducing_points)
        model_x_dict = torch.load(f'models/SVGP/model-x-SpectralMixture-3.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-SpectralMixture-3.pth')
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-SpectralMixture-3.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-SpectralMixture-3.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_SVGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-3',T,save=save_plots)

    # SpectralMixture-5
    if options['SpectralMixture-5']:
        inducing_points = torch.FloatTensor(gp_data[:200]).clone()
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = SVGPModelSpectralMixture_5(inducing_points)
        model_y = SVGPModelSpectralMixture_5(inducing_points)
        model_x_dict = torch.load(f'models/SVGP/model-x-SpectralMixture-5.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-SpectralMixture-5.pth')
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-SpectralMixture-5.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-SpectralMixture-5.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_SVGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-5',T,save=save_plots)

    # SpectralMixture-10
    if options['SpectralMixture-10']:
        inducing_points = torch.FloatTensor(gp_data[:200]).clone()
        test_data = torch.FloatTensor(gp_data)
        test_x_labels = torch.FloatTensor(x_labels)
        test_y_labels = torch.FloatTensor(y_labels)
        likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        model_x = SVGPModelSpectralMixture_10(inducing_points)
        model_y = SVGPModelSpectralMixture_10(inducing_points)
        model_x_dict = torch.load(f'models/SVGP/model-x-SpectralMixture-10.pth')
        model_y_dict = torch.load(f'models/SVGP/model-y-SpectralMixture-10.pth')
        likelihood_x_dict = torch.load(f'models/SVGP/likelihood-x-SpectralMixture-10.pth')
        likelihood_y_dict = torch.load(f'models/SVGP/likelihood-y-SpectralMixture-10.pth')
        model_x.load_state_dict(model_x_dict)
        model_y.load_state_dict(model_y_dict)
        likelihood_x.load_state_dict(likelihood_x_dict)
        likelihood_y.load_state_dict(likelihood_y_dict)
        __test_SVGP(test_data,[test_x_labels,test_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-10',T,save=save_plots)