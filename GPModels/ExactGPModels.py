import gpytorch
import torch

class ExactGPModelRBF(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(ExactGPModelRBF, self).__init__(gp_data, labels, likelihood)
        # Mean Function
        self.mean_module = gpytorch.means.ConstantMean()
        # Covariance Function, i.e. kernel specification
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
		# Compute the mean of the data using the mean function
        mean_x = self.mean_module(x)
        # Compute the covariance of the data using the kernel function
        covar_x = self.covar_module(x)
        # Return a multivariate normal with mean and covariance just computed
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactGPModelRBFPeriodic(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(ExactGPModelRBFPeriodic, self).__init__(gp_data, labels, likelihood)
        # Mean Function
        self.mean_module = gpytorch.means.ConstantMean()
        # Covariance Function, i.e. kernel specification
        self.covar_module = self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()+gpytorch.kernels.PeriodicKernel())

    def forward(self, x):
		# Compute the mean of the data using the mean function
        mean_x = self.mean_module(x)
        # Compute the covariance of the data using the kernel function
        covar_x = self.covar_module(x)
        # Return a multivariate normal with mean and covariance just computed
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactGPModelMatern_32(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(ExactGPModelMatern_32, self).__init__(gp_data, labels, likelihood)
        # Mean Function
        self.mean_module = gpytorch.means.ConstantMean()
        # Covariance Function, i.e. kernel specification
        self.covar_module = self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(1.5))

    def forward(self, x):
		# Compute the mean of the data using the mean function
        mean_x = self.mean_module(x)
        # Compute the covariance of the data using the kernel function
        covar_x = self.covar_module(x)
        # Return a multivariate normal with mean and covariance just computed
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactGPModelMatern_52(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(ExactGPModelMatern_52, self).__init__(gp_data, labels, likelihood)
        # Mean Function
        self.mean_module = gpytorch.means.ConstantMean()
        # Covariance Function, i.e. kernel specification
        self.covar_module = self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(2.5))

    def forward(self, x):
		# Compute the mean of the data using the mean function
        mean_x = self.mean_module(x)
        # Compute the covariance of the data using the kernel function
        covar_x = self.covar_module(x)
        # Return a multivariate normal with mean and covariance just computed
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactGPModelSpectralMixture_3(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(ExactGPModelSpectralMixture_3, self).__init__(gp_data, labels, likelihood)
        # Mean Function
        self.mean_module = gpytorch.means.ConstantMean()
        # Covariance Function, i.e. kernel specification
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=3, ard_num_dims=2)
        # self.covar_module.initialize_from_data(train_data,train_labels)

    def forward(self, x):
		# Compute the mean of the data using the mean function
        mean_x = self.mean_module(x)
        # Compute the covariance of the data using the kernel function
        covar_x = self.covar_module(x)
        # Return a multivariate normal with mean and covariance just computed
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactGPModelSpectralMixture_5(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(ExactGPModelSpectralMixture_5, self).__init__(gp_data, labels, likelihood)
        # Mean Function
        self.mean_module = gpytorch.means.ConstantMean()
        # Covariance Function, i.e. kernel specification
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=5, ard_num_dims=2)
        # self.covar_module.initialize_from_data(train_data,train_labels)

    def forward(self, x):
		# Compute the mean of the data using the mean function
        mean_x = self.mean_module(x)
        # Compute the covariance of the data using the kernel function
        covar_x = self.covar_module(x)
        # Return a multivariate normal with mean and covariance just computed
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactGPModelSpectralMixture_10(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(ExactGPModelSpectralMixture_10, self).__init__(gp_data, labels, likelihood)
        # Mean Function
        self.mean_module = gpytorch.means.ConstantMean()
        # Covariance Function, i.e. kernel specification
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=2)
        # self.covar_module.initialize_from_data(train_data,train_labels)

    def forward(self, x):
		# Compute the mean of the data using the mean function
        mean_x = self.mean_module(x)
        # Compute the covariance of the data using the kernel function
        covar_x = self.covar_module(x)
        # Return a multivariate normal with mean and covariance just computed
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class _FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2,64), # 2 is the input size (x,y)
            torch.nn.ReLU(),
            torch.nn.Linear(64,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,2) # 2 is the output dimension (which is equal to the input dimension)
        )
    def forward(self,x):
        return self.layers(x)
    
class ExactGPModelDeepKernel(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(ExactGPModelDeepKernel, self).__init__(gp_data, labels, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))
        self.feature_extractor = _FeatureExtractor()

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)