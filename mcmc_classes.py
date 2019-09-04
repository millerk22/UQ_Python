import numpy as np
import matplotlib.pyplot as plt
from datasets.util import calc_GR_posterior, threshold1D, threshold2D
from datasets.util import threshold2D_many, threshold2D_avg, threshold1D_avg
import scipy.sparse as sps
from scipy.linalg import eigh
import emcee
import time
import matlab.engine



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

        Note that for the binary case, we have {-1,+1} classes and the entries of self.v_mean
        represent the empirical probability of being class +1 from the samples.
        """
        MCMC_Sampler.run_sampler(self, num_samples)

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




class Gibbs_Probit_Sampler(MCMC_Sampler):
    def __init__(self, gamma=0.01, tau=0., alpha=1.):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)

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
        for i in range(self.Data.N):
            P_KJ[i,i] += self.Data.evals[i]
        #P_KJ = 0.5*(P_KJ + P_KJ.T)  # do we need? seems to be symmetric already...
        S_KJ, Q_KJ = eigh(P_KJ)
        inv_skj = 1./np.sqrt(S_KJ)[:,np.newaxis]

        # Main iterations
        for k in range(burnIn + num_samples):
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
                        self.u_t_mean = threshold2D(u)
                else:
                    self.u_mean = ((k_rec-1) * self.u_mean + u)/k_rec
                    if f == 'thresh':
                        self.u_t_mean = ((k_rec-1) * self.u_t_mean + threshold2D(u))/k_rec

        return









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





"""
print('Testing the get_truncn_samples function')
tic = time.clock()
eng = matlab.engine.start_matlab() # need to put in different place
print('start_matlab took %f' % (time.clock() - tic))


# need to get the object u into the matlab engine
tic = time.clock()
eng.evalc("trand_obj = trandn_multiclass(%s);" % str(data.num_class))
print('trandn_multiclass instantiation took %f' % (time.clock() - tic))

i = 0
tic = time.clock()
u = GR_sampler.m[:100,:].tolist()
eng.workspace['u'] = matlab.double(u)
#command = "s = trand_obj.gen_samples(u, %f, %d);" % (GR_sampler.gamma2, i+1)
#eng.evalc(command, nargout=0)
eng.evalc("s = trand_obj.gen_samples(u, %f, %d);" % (GR_sampler.gamma2, i+1), nargout=0)
s = np.array(eng.workspace['s'])
print('sampling took %f' % (time.clock() - tic))
print(s.shape)
for j in range(s.shape[0]):
    if np.any(s[j,:] > s[j,i]):
        print(s[j,:])
        print(s[j,i])




def ln_truncn(x, mean, gamma2, i):
    if np.any(x) > x[i]:
        return -np.inf
    else:
        diff = x - mean
        if len(mean.shape) > 1:
            nc = mean.shape[1]
            return np.sum([np.inner(diff[:,j], diff[:,j]) for j in range(nc)])/(-2.*gamma2)
        else:
            return np.inner(diff,diff)/(-2.*gamma2)


def get_truncn_samples(u, gamma2, i, num_steps=1000):
    mean = u.flatten()
    Ndim = mean.shape[0]
    Nwalkers = Ndim*3
    S = emcee.EnsembleSampler(Nwalkers, Ndim, ln_truncn, args = (u, gamma2, i))
    positions = emcee.utils.sample_ball(mean, Ndim*[np.sqrt(gamma2)], size=Nwalkers)
    positions, _, _ = S.run_mcmc(positions,num_steps)
    return positions
"""
