import gpytorch
import random

from utils.gp_utils import *
from GPModels.ExactGPModels import *
from GPModels.MultiOutputExactGPModels import *

def train_ExactGP(train_data, x_labels, y_labels):
    idxs = torch.IntTensor(random.sample(range(0,len(train_data)),200))

    # RBF kernel
    train_data = torch.DoubleTensor(train_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelRBF(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelRBF(train_data,train_y_labels,likelihood_y)
    train_ExactGP__(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF')

    # RBF + Periodic kernel
    train_data = torch.DoubleTensor(train_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelRBFPeriodic(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelRBFPeriodic(train_data,train_y_labels,likelihood_y)
    train_ExactGP__(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF-Periodic')

    # Matern 3/2 kernel
    train_data = torch.DoubleTensor(train_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelMatern_32(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelMatern_32(train_data,train_y_labels,likelihood_y)
    train_ExactGP__(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-32')

    # Matern 5/2 kernel
    train_data = torch.DoubleTensor(train_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelMatern_52(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelMatern_52(train_data,train_y_labels,likelihood_y)
    train_ExactGP__(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-52')

    # Spectral Mixture (n=4) kernel
    train_data = torch.DoubleTensor(train_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelSpectralMixture_3(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelSpectralMixture_3(train_data,train_y_labels,likelihood_y)
    train_ExactGP__(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-3')

def train_MultiOutputExactGP(train_data, x_labels, y_labels):
    '''
    Labels are of the form [x_label, y_label]
    '''
    idxs = torch.IntTensor(random.sample(range(0,len(train_data)),200))

    # RBF
    train_data = torch.DoubleTensor(train_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    ## Stack labels to get single 2-dimensional label
    train_labels = torch.stack([train_x_labels,train_y_labels],dim=1)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultiOutputExactGPModelRBF(train_data, train_labels, likelihood)
    train_ExactMultiOutputExactGP__(train_data,train_labels,model,likelihood,'RBF')
