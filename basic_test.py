from mcmc_classes import *
from datasets.dataloaders import *
from util.util import *
import numpy as np
from IPython import embed
import sys
import time
import argparse


if __name__ == "__main__":

    # Test script command line parameter parsing
    show_plot = True
    run_ipy = False
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('--show')
        parser.add_argument('--embed')
        args = parser.parse_args()
        if args.show is not None:
            show_plot = int(args.show)
        if args.embed is not None:
            run_ipy = int(args.embed)



    print("\nRunning Gaussian Cluster test of MCMC sampler, Gibbs Probit")
    # load Gaussian Clusters data, plot initial distribution of labeled and unlabeled
    Ns = [100,200,100]
    mus = [np.array([1., 0.]), np.array([-1., 0.]), np.array([0., 1.5])]
    sigma = 0.15
    Covs = [sigma*np.eye(2) for j in range(len(mus))]
    data = load_gaussian_cluster(Ns, mus, Covs)
    if show_plot:
        data.plot_initial()

    # Create the Gibbs-Probit sampler, default params : gamma = 0.01, tau = 0.01, alpha = 1.0
    GPS = Gibbs_Probit_Sampler()
    GPS.load_data(data)
    # Get 1000 samples (i.e. calculate the posterior mean and covariance, then sample)
    num_samples = 1000
    tic = time.process_time()
    GPS.run_sampler(num_samples)
    print('%s :\n\tSampling of %d samples took %f seconds ' % (str(GPS), num_samples, time.process_time() - tic))
    GPS.comp_mcmc_stats(return_acc=False)
    print("\tAccuracy of u_mean: %f" % GPS.sum_stats.acc)

    # embed the current namespace in an iPython session in the terminal
    if run_ipy:
        embed()
