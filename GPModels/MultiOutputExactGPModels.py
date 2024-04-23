import gpytorch

class MultiOutputExactGPModelRBF(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(MultiOutputExactGPModelRBF, self).__init__(gp_data, labels, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class MultiOutputExactGPModelRBFPeriodic(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(MultiOutputExactGPModelRBFPeriodic, self).__init__(gp_data, labels, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel()+gpytorch.kernels.PeriodicKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class MultiOutputExactGPModelMatern_32(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(MultiOutputExactGPModelMatern_32, self).__init__(gp_data, labels, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(1.5), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class MultiOutputExactGPModelMatern_52(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(MultiOutputExactGPModelMatern_52, self).__init__(gp_data, labels, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(2.5), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class MultiOutputExactGPModelSpectralMixture_3(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(MultiOutputExactGPModelSpectralMixture_3, self).__init__(gp_data, labels, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.SpectralMixtureKernel(num_mixtures=3,ard_num_dims=2), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class MultiOutputExactGPModelSpectralMixture_5(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(MultiOutputExactGPModelSpectralMixture_5, self).__init__(gp_data, labels, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.SpectralMixtureKernel(num_mixtures=5,ard_num_dims=2), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class MultiOutputExactGPModelSpectralMixture_10(gpytorch.models.ExactGP):
    def __init__(self, gp_data, labels, likelihood):
        super(MultiOutputExactGPModelSpectralMixture_10, self).__init__(gp_data, labels, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10,ard_num_dims=2), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
