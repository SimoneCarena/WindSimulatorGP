import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # Mean Function
        self.mean_module = gpytorch.means.ConstantMean()
        # Covariance Function, i.e. kernel specification
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))

    def forward(self, x):
		# Compute the mean of the data using the mean function
        mean_x = self.mean_module(x)
        # Compute the covariance of the data using the kernel function
        covar_x = self.covar_module(x)
        # Return a multivariate normal with mean and covariance just computed
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)