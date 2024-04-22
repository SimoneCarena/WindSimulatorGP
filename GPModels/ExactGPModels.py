import gpytorch

class ExactGPModelRBF(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(ExactGPModelRBF, self).__init__(gp_data, labels, likelihood)
        # Mean Function
        self.mean_module = gpytorch.means.ConstantMean()
        # Covariance Function, i.e. kernel specification
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.double()

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
        self.double()

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
        self.double()

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
        self.double()

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
        self.double()

    def forward(self, x):
		# Compute the mean of the data using the mean function
        mean_x = self.mean_module(x)
        # Compute the covariance of the data using the kernel function
        covar_x = self.covar_module(x)
        # Return a multivariate normal with mean and covariance just computed
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
