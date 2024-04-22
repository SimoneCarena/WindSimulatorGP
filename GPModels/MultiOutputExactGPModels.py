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
        self.double()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)