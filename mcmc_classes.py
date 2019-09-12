import numpy as np
import matplotlib.pyplot as plt
from util.util import calc_GR_posterior, threshold1D, threshold2D
from util.util import threshold2D_avg, threshold1D_avg, SummaryStats
import scipy.sparse as sps
from scipy.linalg import eigh
from scipy.stats import norm
import emcee
import time
import matlab.engine
from datasets.trnm import TruncRandNormMulticlass


class MCMC_Sampler(object):
    def __init__(self, gamma, tau=0., alpha=1.):
        self.gamma2 = gamma**2.
        self.tau = tau
        self.alpha = alpha
        self.stats = None
        self.Data = None
        self.sum_stats = SummaryStats()
        self.sum_stats_t = SummaryStats()

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
        self.print_ = False
        if num_samples+burnIn > 1000:
            self.print_ = True


    def comp_mcmc_stats(self):
        print("\tComputing summary statistics...")

        if self.Data is None:
            raise ValueError('No Data object loaded yet...')
        if self.u_mean is None:
            raise ValueError('Have not sampled yet... need to run sampler to obtain stats.')

        # Get class labelings of computed means, for use in summary stats computation
        # (this gives a 1D np array with computed class labeling)
        if -1 in self.Data.classes:
            u_mean_t = threshold1D(self.u_mean.copy())
            u_t_mean_t = threshold1D(self.u_t_mean.copy())
        elif 0 in self.Data.classes and len(self.u_mean.shape) == 1:
            u_mean_t = threshold1D(self.u_mean.copy(), True)
            u_t_mean_t = threshold1D(self.u_t_mean.copy(), True)
        else:
            u_mean_t = threshold2D(self.u_mean.copy(), False)
            u_t_mean_t = threshold2D(self.u_t_mean.copy(), False)

        self.sum_stats.compute(self.Data.ground_truth, u_mean_t, self.Data.N, self.Data.num_class)
        self.sum_stats_t.compute(self.Data.ground_truth, u_t_mean_t, self.Data.N, self.Data.num_class)

        return self.sum_stats.acc, self.sum_stats_t.acc


    def active_learning_choices(self, method, num_to_label):
        pass

    def plot_u(self, u):
        #assert len(self.Data.classes) == 2
        plt.scatter(np.arange(self.Data.N), u)
        plt.scatter(self.Data.labeled, u[self.Data.labeled], c='g')
        plt.title('Plot of u')
        plt.show()


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

        Note that for the binary case, we have {-1,+1} classes and the entries of self.v_mean
        represent the empirical probability of being class +1 from the samples.
        """
        MCMC_Sampler.run_sampler(self, num_samples)
        print("Running Gaussian Regression sampling to get %d samples, with no burnIn samples" % num_samples)
        ## run Gaussian Regression method here -- ignoring burnIn
        self.m, self.C, self.y = calc_GR_posterior(self.Data.evecs, self.Data.evals, self.Data.fid,
                                self.Data.labeled, self.Data.unlabeled, self.tau, self.alpha, self.gamma2)

        # binary class case
        if self.Data.num_class == 2:
            samples = np.random.multivariate_normal(self.m, self.C, num_samples).T
            self.u_mean = np.average(samples, axis=1)
            if f == 'thresh':
                self.u_t_mean = threshold1D_avg(samples)

        # multiclass sampling case
        else:
            samples = np.array([np.random.multivariate_normal(self.m[:,i], self.C,
                            self.num_samples).T for i in range(self.Data.num_class)]).transpose((1,0,2))
            self.u_mean = np.average(samples, axis=2)
            if f == 'thresh':
                self.u_t_mean = threshold2D_avg(samples)

        # delete the samples for sake of memory
        del samples
        return

    def comp_mcmc_stats(self):
        self.acc_u, self.acc_u_t = MCMC_Sampler.comp_mcmc_stats(self)
        """ In addition to the stats from the sampling from the posterior, calculate
        accuracy of thresholded analytic posterior mean, threshold*D(self.m)"""
        if -1 in self.Data.fid.keys():
            m_t = threshold1D(self.m)
        else:
            m_t = threshold2D(self.m, False)
        self.acc_m = len(np.where(m_t == self.Data.ground_truth)[0])/self.Data.N
        return self.acc_u, self.acc_u_t



class Gibbs_Probit_Sampler(MCMC_Sampler):
    def __init__(self, gamma=0.01, tau=0., alpha=1.):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)
        self.gamma = gamma
        if tau != 0. or alpha != 1.:
            print('Sampling for tau != 0. and alpha != 1. is not yet implemented, proceeding with default values')

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)
        if -1 in self.Data.fid.keys():
            print("Noticed you gave Gibbs-Probit classes with -1. Converting to 0 for the current implementation...")
            self.Data.fid[0] = self.Data.fid[-1]
            del self.Data.fid[-1]
            self.Data.ground_truth[self.Data.ground_truth == -1] = 0.
            self.Data.classes = [0,1]
        # initiate the trandn_multiclass object
        print("Gibbs Probit Sampler using Python functions.")
        self.TRNM = TruncRandNormMulticlass(self.Data.num_class)


    def run_sampler(self, num_samples, burnIn=0, f='thresh'):
        MCMC_Sampler.run_sampler(self, num_samples, burnIn)
        print("Running Gibbs-Probit sampling to get %d samples, with %d burnIn samples" % (num_samples, burnIn))

        # instantiate the u_mean (unthresh), u_t_mean (unthres) variables
        self.u_mean = np.zeros((self.Data.N, self.Data.num_class))
        self.u_t_mean = np.zeros_like(self.u_mean)

        # fixed initialization of u.
        u = np.zeros_like(self.u_mean)

        # fidv contains the indices of the fidelity nodes in the list "labeled"
        fidv = {}
        for c, ind in self.Data.fid.items():
            u[np.ix_(ind, len(ind)*[c])] = 1.
            fidv[c] = [self.Data.labeled.index(j) for j in ind]

        # sample Gaussian noise in batch to begin
        z_all = np.random.randn(len(self.Data.evals), self.Data.num_class, num_samples+burnIn)

        # Compute the projections for use in the KL expansion for sampling u | v
        V_KJ = self.Data.evecs[self.Data.labeled,:]
        P_KJ = V_KJ.T.dot(V_KJ)/self.gamma2
        for i in range(len(self.Data.evals)):
            P_KJ[i,i] += self.Data.evals[i]
        #P_KJ = 0.5*(P_KJ + P_KJ.T)  # do we need? seems to be symmetric already...
        S_KJ, Q_KJ = eigh(P_KJ)
        inv_skj = 1./np.sqrt(S_KJ)[:,np.newaxis]


        # Main iterations
        self.U = []
        for k in range(burnIn + num_samples):
            if self.print_ and k % 500 == 0:
                print('\tIteration %d of sampling...' % k)
            z_k = z_all[:,:,k] # the white noise samples for use in sampling u | v

            # Sample v ~ P(v | u). Uses the MATLAB function for trandn_multiclass MATLAB object
            v = np.zeros((len(self.Data.labeled), self.Data.num_class))

            for cl, ind_cl in self.Data.fid.items():
                v[fidv[cl],:] = self.TRNM.gen_samples(u[ind_cl,:], self.gamma, cl)



            # Sample u ~ P(u | v) via KL expansion
            temp = V_KJ.T.dot(v)/self.gamma2
            temp = Q_KJ.T.dot(temp)
            temp /= S_KJ[:,np.newaxis]
            m_hat = Q_KJ.dot(temp)
            u_hat = Q_KJ.dot(z_k * inv_skj) + m_hat
            u = self.Data.evecs.dot(u_hat)


            # If past the burn in period, calculate the updates to the means we're recording
            if k > burnIn:
                self.U.append(u)
                k_rec = k - burnIn
                if k_rec == 1:
                    self.u_mean = u
                    if f == 'thresh':
                        self.u_t_mean = threshold2D(u.copy())
                else:
                    self.u_mean = ((k_rec-1) * self.u_mean + u)/k_rec
                    if f == 'thresh':
                        self.u_t_mean = ((k_rec-1) * self.u_t_mean + threshold2D(u.copy()))/k_rec

        return





class pCN_Probit_Sampler(MCMC_Sampler):
    def __init__(self, beta, gamma=0.1, tau=0., alpha=1.):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)
        self.gamma = gamma
        self.beta = beta

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)
        if max(self.Data.fid.keys()) > 1:
            raise NotImplementedError("Multiclass sampling for pCN Probit is not yet implemented")

        # self.y has ordering as given in variable self.labeled
        ofs = min(self.Data.fid.keys())
        self.y = np.ones(len(self.Data.labeled))
        mask = [np.where(self.Data.labeled == v)[0] for v in self.Data.fid[ofs]]
        self.y[mask] = ofs


    def run_sampler(self, num_samples, burnIn=0, f='thresh'):
        MCMC_Sampler.run_sampler(self, num_samples, burnIn)
        print("Running pCN Probit sampling to get %d samples, with %d burnIn samples" % (num_samples, burnIn))

        norm_rv = norm(scale=self.gamma)  # normal rv for generating the cdf values

        # Helper functions for running the MH sampling
        def log_like(x):
            return -np.sum(np.log(norm_rv.cdf(x * self.y)))

        def alpha(u, w):
            u_, w_ = u[self.Data.labeled], w[self.Data.labeled]

            return np.min([1., np.exp(log_like(u_) - log_like(w_))])

        # Sample Gaussian noise in batch


        if self.tau > 0:
            KL_scaled_evecs =  self.Data.evecs * self.Data.evals**-0.5
            z = np.random.randn(self.Data.evals.shape[0], num_samples+burnIn)
        else:
            KL_scaled_evecs =  self.Data.evecs[:,1:] * self.Data.evals[1:]**-0.5
            z = np.random.randn(self.Data.evals.shape[0]-1, num_samples+burnIn)

        # instantiate sample
        u = np.ones(self.Data.N, dtype=np.float32)*np.average(list(self.Data.fid.keys()))
        for v in self.Data.fid.keys():
            u[self.Data.fid[v]] = v



        # Main iterations
        for k in range(burnIn + num_samples):
            if self.print_ and k % 500 == 0:
                print('\tIteration %d of sampling...' % k)
            # proposal step
            w_k = np.sqrt(1. - self.beta**2.)*u + KL_scaled_evecs.dot(z[:,k])*self.beta

            # calc acceptance prob, and accept/reject proposal step accordingly
            acc_prob = alpha(u, w_k)
            if np.random.rand() <= acc_prob:
                u = w_k

            # Record mean if past burn-in stage
            if k >= burnIn:
                k_rec = k - burnIn
                if k_rec == 0:
                    self.u_mean = u
                    if f == 'thresh':
                        self.u_t_mean = threshold1D(u.copy())
                else:
                    self.u_mean = ((k_rec) * self.u_mean + u)/(k_rec + 1)
                    if f == 'thresh':
                        self.u_t_mean = ((k_rec) * self.u_t_mean + threshold1D(u.copy()))/(k_rec+1)






class pCN_BLS_Sampler(MCMC_Sampler):
    def __init__(self, beta, gamma=0.1, tau=0., alpha=1.):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)
        self.gamma = gamma
        self.beta = beta

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)
        if max(self.Data.fid.keys()) > 1:
            raise NotImplementedError("Multiclass sampling for pCN BLS is not yet implemented")

        # self.y has ordering as given in variable self.labeled
        self.y = self.Data.ground_truth[self.Data.labeled]
        self.zero_one = False
        if 0 in self.Data.fid.keys():
            self.zero_one = True
        print("classes contains 0 is %s" % str(self.zero_one))

    def run_sampler(self, num_samples, burnIn=0, f='thresh'):
        MCMC_Sampler.run_sampler(self, num_samples, burnIn)
        print("Running pCN BLS sampling to get %d samples, with %d burnIn samples" % (num_samples, burnIn))

        norm_rv = norm(scale=self.gamma)  # normal rv for generating the cdf values

        # Helper functions for running the MH sampling
        def log_like(x):
            return len(np.where(x != self.y)[0])/(np.sqrt(2.)*self.gamma2)

        def alpha(u, w):
            u_, w_ = threshold1D(u[self.Data.labeled].copy(), self.zero_one), threshold1D(w[self.Data.labeled].copy(), self.zero_one)
            return np.min([1., np.exp(log_like(u_) - log_like(w_))])

        # Sample Gaussian noise in batch
        if self.tau > 0:
            KL_scaled_evecs =  self.Data.evecs * self.Data.evals**-0.5
            z = np.random.randn(self.Data.evals.shape[0], num_samples+burnIn)
        else:
            KL_scaled_evecs =  self.Data.evecs[:,1:] * self.Data.evals[1:]**-0.5
            z = np.random.randn(self.Data.evals.shape[0]-1, num_samples+burnIn)

        # instantiate sample
        u = np.ones(self.Data.N, dtype=np.float32)*np.average(list(self.Data.fid.keys()))
        for v in self.Data.fid.keys():
            u[self.Data.fid[v]] = v

        # Main iterations
        self.accepted = 0
        for k in range(burnIn + num_samples):
            if self.print_ and k % 500 == 0:
                print('\tIteration %d of sampling...' % k)
            # proposal step
            w_k = np.sqrt(1. - self.beta**2.)*u + KL_scaled_evecs.dot(z[:,k])*self.beta


            # calc acceptance prob, and accept/reject the proposal step accordingly
            acc_prob = alpha(u, w_k)
            if np.random.rand() <= acc_prob:
                u = w_k
                self.accepted += 1

            # Record mean if past burn-in stage
            if k >= burnIn:
                k_rec = k - burnIn
                if k_rec == 0:
                    self.u_mean = u.copy()
                    if f == 'thresh':
                        self.u_t_mean = threshold1D(u.copy(), self.zero_one)
                else:
                    self.u_mean = ((k_rec) * self.u_mean + u)/(k_rec + 1)
                    if f == 'thresh':
                        self.u_t_mean = ((k_rec) * self.u_t_mean + threshold1D(u.copy(), self.zero_one))/(k_rec+1)






""" OLD MATLAB based probit gibbs sampler

class Gibbs_Probit_Sampler_MATLAB(MCMC_Sampler):
    def __init__(self, gamma=0.01, tau=0., alpha=1.):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)
        if tau != 0. or alpha != 1.:
            print('Sampling for tau != 0. and alpha != 1. is not yet implemented, proceeding with default values')

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)

        # initiate the trandn_multiclass object
        print("Gibbs Probit Sampler's current implementation uses MATLAB functions...")
        print('\tInstantiating the matlab objects for trandn_multiclass sampler')
        tic = time.process_time()
        self.eng = matlab.engine.start_matlab()
        print('\tmatlab engine instantiation took %s seconds' % str(time.process_time() - tic ))
        self.eng.evalc("trand_obj = trandn_multiclass(%s);" % str(Data.num_class))
        self.eng.evalc("fid = {};")
        for c, ind in self.Data.fid.items():
            self.eng.evalc("fid{%d} = %s;"%(c+1,str([a+ 1 for a in ind])))
        print('\tFinished MATLAB initialization')


    def run_sampler(self, num_samples, burnIn=0, f='thresh'):
        MCMC_Sampler.run_sampler(self, num_samples, burnIn)

        # instantiate the u_mean (unthresh), u_t_mean (unthres) variables
        self.u_mean = np.zeros((self.Data.N, self.Data.num_class))
        self.u_t_mean = np.zeros_like(self.u_mean)

        # fixed initialization of u.
        u = np.zeros_like(self.u_mean)

        # fidv contains the indices of the fidelity nodes in the list "labeled"
        fidv = {}
        for c, ind in self.Data.fid.items():
            u[np.ix_(ind, len(ind)*[c])] = 1.
            fidv[c] = [self.Data.labeled.index(j) for j in ind]

        # sample Gaussian noise in batch to begin
        z_all = np.random.randn(len(self.Data.evals), self.Data.num_class, num_samples+burnIn)

        # Compute the projections for use in the KL expansion for sampling u | v
        V_KJ = self.Data.evecs[self.Data.labeled,:]
        P_KJ = (1./self.gamma2)*V_KJ.T.dot(V_KJ)
        for i in range(len(self.Data.evals)):
            P_KJ[i,i] += self.Data.evals[i]
        #P_KJ = 0.5*(P_KJ + P_KJ.T)  # do we need? seems to be symmetric already...
        S_KJ, Q_KJ = eigh(P_KJ)
        inv_skj = 1./np.sqrt(S_KJ)[:,np.newaxis]

        # Main iterations
        for k in range(burnIn + num_samples):
            if self.print_ and k % 500 == 0:
                print('\tIteration %d of sampling...' % k)
            z_k = z_all[:,:,k] # the white noise samples for use in sampling u | v

            # Sample v ~ P(v | u). Uses the MATLAB function for trandn_multiclass MATLAB object
            v = np.zeros((len(self.Data.labeled), self.Data.num_class))
            #tic = time.process_time()
            for cl, ind_cl in self.Data.fid.items():
                self.eng.workspace['u_cl'] = matlab.double(u[ind_cl,:].tolist())
                self.eng.evalc("v_cl = trand_obj.gen_samples(u_cl, %f, %d);" % (self.gamma2**0.5, cl+1))
                v[fidv[cl],:] = np.asarray(self.eng.workspace['v_cl'])


            # Sample u ~ P(u | v) via KL expansion
            temp = V_KJ.T.dot(v)/self.gamma2
            temp = Q_KJ.T.dot(temp)
            temp *= inv_skj
            m_hat = Q_KJ.dot(temp)
            u_hat = Q_KJ.dot(z_k * inv_skj) + m_hat
            u = self.Data.evecs.dot(u_hat)

            # If past the burn in period, calculate the updates to the means we're recording
            if k > burnIn:
                k_rec = k - burnIn
                if k_rec == 1:
                    self.u_mean = u
                    if f == 'thresh':
                        self.u_t_mean = threshold2D(u.copy())
                else:
                    self.u_mean = ((k_rec-1) * self.u_mean + u)/k_rec
                    if f == 'thresh':
                        self.u_t_mean = ((k_rec-1) * self.u_t_mean + threshold2D(u.copy()))/k_rec

        return

"""
