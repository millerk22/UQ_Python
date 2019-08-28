import numpy as np
import matplotlib.pyplot as plt
from datasets.util import calc_orig_multi, threshold1D, threshold2D
from datasets.util import threshold2D_many, threshold2D_avg, threshold1D_avg


class MCMC_Sampler(object):
    def __init__(self, gamma, tau=0., alpha=1.):
        self.gamma2 = gamma**2.
        self.tau = tau
        self.alpha = alpha
        self.stats = None
        self.Data = None

    def load_data(self, Data, plot_=False):
        self.Data = Data
        if not self.Data.have_useful:
            self.Data.get_useful_structs()
        if plot_:
            self.Data.plot_initial()

    def run_sampler(self, num_samples, burnIn=0):
        if self.Data is None:
            raise ValueError("No data has been loaded. First load data via member function load_data(Data)")
        self.num_samples = num_samples
        self.burnIn = burnIn


    def comp_mcmc_stats(self):
        # Confusion matrix, recall, precision, recall_conf, precision_conf, acc
        pass


    def active_learning_choices(self, method, num_to_label):
        pass


class Gaussian_Regression_Sampler(MCMC_Sampler):
    def __init__(self, gamma=0.001, tau=0.01, alpha=1.0):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)


    def run_sampler(self, num_samples, f='thresh'):
        """
        Run MCMC sampling for the loaded dataset.
        Inputs:
            num_samples : int , number of desired samples from this sampling.
                (Note for GR here no burnIn needed)
            f : str 'thresh' or function handle, samples will compute values related to
                E[f(u)]. On

        Outputs:
            Saves to the Sampler object:
                self.samples : (N x num_class x num_samples) numpy array of pre-thresholded samples
                self.u_mean : (N x num_class) numpy array of pre-thresholded sample mean
                self.v_mean : (N x num_class) numpy array of thresholded sample mean
                self.m : (N x num_class) numpy array analytical mean (GR-special)
                self.C : (N x N) numpy array analytical covariance operator (GR-special)
                self.y : (N' x num_class) numpy array of labelings on labeled set
        """
        MCMC_Sampler.run_sampler(self, num_samples)

        ## run Gaussian Regression method here -- ignoring burnIn
        self.m, self.C, self.y = calc_orig_multi(self.Data.evecs, self.Data.evals, self.Data.fid,
                                self.Data.labeled, self.Data.unlabeled, self.tau, self.alpha, self.gamma2)
        if self.Data.num_class == 2:
            samples = np.random.multivariate_normal(self.m, self.C, num_samples).T
            self.u_mean = np.average(samples, axis=1)
            if f == 'thresh':
                self.v_mean = threshold1D_avg(samples)

        else:
            samples = np.array([np.random.multivariate_normal(self.m[:,i], self.C,
                            self.num_samples).T for i in range(self.Data.num_class)]).transpose((1,0,2))
            self.u_mean = np.average(samples, axis=2)
            if f == 'thresh':
                self.v_mean = threshold2D_avg(samples)

        del samples




class Gibbs_Probit_Sampler(MCMC_Sampler):
    def __init__(self, gamma, tau=0., alpha=1.):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)

    def run_sampler(self, num_samples, burnIn=0):
        MCMC_Sampler.run_sampler(self, num_samples, burnIn)
        ## run Gibbs Probit method here




class pCN_Probit_Sampler(MCMC_Sampler):
    def __init__(self, gamma, tau=0., alpha=1.):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)

    def run_sampler(self, num_samples, burnIn=0):
        MCMC_Sampler.run_sampler(self, num_samples, burnIn)
        ## run pCN Probit method here






class pCN_BLS_Sampler(MCMC_Sampler):
    def __init__(self, gamma, tau=0., alpha=1.0):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)

    def run_sampler(self, num_samples, burnIn=0):
        MCMC_Sampler.run_sampler(self, num_samples, burnIn)

        ## run pCN BLS method here
