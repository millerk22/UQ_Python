from util.util import threshold1D, threshold2D
class MCMC(object):
    def __init__(self, w, v, fid, params):
        """
        w : (k,  ) eigenvalues of the graph laplacian
        v : (N, k) eigenvectors of the graph laplacian

        fid: dictionary, fidelity points
        params:
            alpha
            tau
            gamma
            seed
        """
        # noise variance
        self.gamma  = params['gamma']
        self.gamma2 = params['gamma'] ** 2
        self.w      = (w + params['tau'] ** 2) ** params['alpha'] 
        self.v      = v
        self.fid    = fid
        self.seed = params['seed']

        # self.n : number of nodes
        self.n = v.shape[0]
        self.classes = list(fid.keys())
        self.n_classes = len(self.classes)
        self.funcs = [threshold2D]

        self.samples = []
        self.n_samples = 0 
        # self.means[func] = E [func(u)] 
        self.means = dict()
        self.burnin = 0

    def set_burnin(self, burnin):
        self.burnin = burnin

    def add_funcs(self, funcs):
        """
        Track the posterior mean of func[u] for each func in funcs
        """
        for func in funcs:
            if func not in self.funcs:
                self.funcs.append(func)

    def compute_means(self, funcs):
        for func in funcs:
            if func in self.means:
                yield self.means[func]
            else:
                if not self.samples or len(self.samples) <= self.burnin:
                    raise ValueError('No samples or not enough samples')
                s = self.samples[0] * 0
                for i, u in enumerate(self.samples):
                    if i >= self.burnin:
                        s += func(u)
                s /= (self.n_samples - self.burnin) 
                self.means[func] = s
                yield self.means[func]
    



    
