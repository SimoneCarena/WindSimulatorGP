import gpytorch
import random

from utils.gp_utils import *
from GPModels.ExactGPModels import *
from GPModels.MultiOutputExactGPModels import *

def train_ExactGP(gp_data, x_labels, y_labels,training_iter=10000):
    idxs = torch.IntTensor(random.sample(range(0,len(gp_data)),200))

    # RBF kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelRBF(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelRBF(train_data,train_y_labels,likelihood_y)
    train_ExactGP__(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF',training_iter)

    # RBF + Periodic kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelRBFPeriodic(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelRBFPeriodic(train_data,train_y_labels,likelihood_y)
    train_ExactGP__(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'RBF-Periodic',training_iter)

    # Matern 3/2 kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelMatern_32(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelMatern_32(train_data,train_y_labels,likelihood_y)
    train_ExactGP__(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-32',training_iter)

    # Matern 5/2 kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelMatern_52(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelMatern_52(train_data,train_y_labels,likelihood_y)
    train_ExactGP__(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'Matern-52',training_iter)

    # Spectral Mixture (n=4) kernel
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
    model_x = ExactGPModelSpectralMixture_3(train_data,train_x_labels,likelihood_x)
    model_y = ExactGPModelSpectralMixture_3(train_data,train_y_labels,likelihood_y)
    train_ExactGP__(train_data,[train_x_labels,train_y_labels],[model_x,model_y],[likelihood_x,likelihood_y],'SpectralMixture-3',training_iter)

def train_MultiOutputExactGP(gp_data, x_labels, y_labels,training_iter=10000):
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
    train_ExactMultiOutputExactGP__(train_data,train_labels,model,likelihood,'RBF',training_iter)

    # RBF+Periodic
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    ## Stack labels to get single 2-dimensional label
    train_labels = torch.stack([train_x_labels,train_y_labels],dim=1)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultiOutputExactGPModelRBFPeriodic(train_data, train_labels, likelihood)
    train_ExactMultiOutputExactGP__(train_data,train_labels,model,likelihood,'RBF-Periodic',training_iter)

    # Matern 3/2
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    ## Stack labels to get single 2-dimensional label
    train_labels = torch.stack([train_x_labels,train_y_labels],dim=1)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultiOutputExactGPModelMatern_32(train_data, train_labels, likelihood)
    train_ExactMultiOutputExactGP__(train_data,train_labels,model,likelihood,'Matern-32',training_iter)

    # Matern 5/2
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    ## Stack labels to get single 2-dimensional label
    train_labels = torch.stack([train_x_labels,train_y_labels],dim=1)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultiOutputExactGPModelMatern_52(train_data, train_labels, likelihood)
    train_ExactMultiOutputExactGP__(train_data,train_labels,model,likelihood,'Matern-52',training_iter)

    # Specrtal mixture 3
    train_data = torch.DoubleTensor(gp_data).index_select(0,idxs).clone()
    train_x_labels = torch.DoubleTensor(x_labels).index_select(0,idxs).clone()
    train_y_labels = torch.DoubleTensor(y_labels).index_select(0,idxs).clone()
    ## Stack labels to get single 2-dimensional label
    train_labels = torch.stack([train_x_labels,train_y_labels],dim=1)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultiOutputExactGPModelSpectralMixture_3(train_data, train_labels, likelihood)
    train_ExactMultiOutputExactGP__(train_data,train_labels,model,likelihood,'SpectralMixture-3',training_iter)
