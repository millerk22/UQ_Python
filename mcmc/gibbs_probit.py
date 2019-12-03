from .mcmc import MCMC
import numpy as np
from scipy.linalg import eigh
from datasets.trnm import TruncRandNormMulticlass
class Gibbs_probit_sampler(MCMC):
    def __init__(self, w, v, fid, params):
        MCMC.__init__(self, w, v, fid, params)
        self.TRNM = TruncRandNormMulticlass(self.n_classes)

    def sample(self, n_samples, u0 = None, save_samples=False):
        """
        generate additional n_samples
        
        u0 : initial sample
        save_samples: whether or not we should save the samples
        """
        # if we have previous samples, we use the last sample
        # otherwise, we pick a new one
        if u0:
            u = u0
        elif self.samples:
            u = self.samples[-1]
        else:
            u = np.zeros(self.n, self.n_classes)

        # enforce fidelity points
        labeled = []
        for c, ind in self.fid.items():
            u[ind, len(ind)*[c]] = 1.
            labeled += ind

        # sample Gaussian noise in batch to begin
        np.random.seed(self.seed)
        z_all = np.random.randn(len(self.w), self.n_classes, n_samples)

        # Compute the projections for use in the KL expansion for sampling u | v

        V_KJ = self.v[labeled,:]
        P_KJ = V_KJ.T.dot(V_KJ)/self.gamma2
        for i in range(len(self.w)):
            P_KJ[i,i] += self.w[i]
        #P_KJ = 0.5*(P_KJ + P_KJ.T)  # do we need? seems to be symmetric already...
        S_KJ, Q_KJ = eigh(P_KJ)
        inv_skj = 1./np.sqrt(S_KJ)[:,np.newaxis]


        # Main iterations
        for i in range(n_samples):
            if  i % 1000 == 0:
                print('\tIteration %d of sampling...' % k)
            z_k = z_all[:,:,i] # the white noise samples for use in sampling u | v
            # Sample v ~ P(v | u).
            v = []
            for c, ind in self.fid.items():
                v += self.TRNM.gen_samples(u[ind, :], self.gamma)
            v = np.vstack(v)

            # Sample u ~ P(u | v) via KL expansion
            temp = V_KJ.T.dot(v)/self.gamma2
            temp = Q_KJ.T.dot(temp)
            temp /= S_KJ[:,np.newaxis]
            m_hat = Q_KJ.dot(temp)
            u_hat = Q_KJ.dot(z_k * inv_skj) + m_hat
            u = self.v.dot(u_hat)


            # If past the burn in period, calculate the updates to the means we're recording
            if save_samples:
                self.samples.append(u)

            self.n_samples += 1
            k_rec = self.n_samples - self.burnin
            if k_rec == 1:
                for func in self.funcs:
                    self.means[func] = func(u)
            else:
                for func in self.funcs:
                    self.means[func] = ((k_rec - 1) * self.means[func] \
                                     + func(u)) / k_rec
        return u # return the last u
