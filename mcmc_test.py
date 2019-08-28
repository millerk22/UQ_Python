from mcmc_classes import *
from datasets.dataloaders import *
from datasets.util import *
import numpy as np
from IPython import embed


if __name__ == "__main__":
    print("Running test of MCMC sampler, Gaussian Regression")

    """
    # load the 2 moons data, plot initial distribution of labeled and unlabeled
    data = load_2_moons()
    data.plot_initial()

    # Create the GR sampler, default params : gamma = 0.001, tau = 0.01, alpha = 1.0
    GR_sampler = Gaussian_Regression_Sampler()
    GR_sampler.load_data(data)

    # Get 100 samples (i.e. calculate the posterior mean and covariance, then sample)
    GR_sampler.run_sampler(10000)
    print(GR_sampler.u_mean.shape)
    print(GR_sampler.v_mean.shape)
    """



    # load the 2 moons data, plot initial distribution of labeled and unlabeled
    Ns = [100,200,100]
    mus = [np.array([1., 0.]), np.array([-1., 0.]), np.array([1., 1.5])]
    sigma = 0.2
    Covs = [sigma*np.eye(2) for j in range(len(mus))]
    data = load_gaussian_cluster(Ns, mus, Covs)
    data.plot_initial()

    # Create the GR sampler, default params : gamma = 0.001, tau = 0.01, alpha = 1.0
    GR_sampler = Gaussian_Regression_Sampler()
    GR_sampler.load_data(data)

    # Get 100 samples (i.e. calculate the posterior mean and covariance, then sample)
    GR_sampler.run_sampler(100)
    print(GR_sampler.u_mean[:5,:])
    print(GR_sampler.v_mean[:5,:])
    embed()
