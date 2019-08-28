import numpy as np
import scipy.sparse as sps
import scipy.linalg as sla
from sklearn.datasets import make_moons
from .util import plot_iter_multi, plot_iter, calc_stats_multi


## TODO: Need a way to have an "original" fid dictionary, that can then pass to
# different runs of Active Learning with UQ samples.
class Data_obj(object):
    def __init__(self, X, evals, evecs, fid, ground_truth):
        self.X = X
        self.evals = evals
        self.evecs = evecs
        self.fid = fid
        self.ground_truth = ground_truth
        self.N = self.X.shape[0]
        self.classes = list(self.fid.keys())
        self.num_class = len(self.classes)
        self.have_useful = False

    def get_useful_structs(self):
        self.org_indices = []  # org_indices organizes indices of nodes according to ground truth class assignments
        for i in self.classes:
            self.org_indices.extend(list(np.where(self.ground_truth == i)[0]))

        indices = np.array(list(range(self.N)))  # 0, 1, 2, ... for use in other calculations
        self.labeled = set()        # labeled nodes list
        self.gt_flipped = {}      # ground_truth "flipped" k,v pairs -> i : class(i)
        for i in self.classes:
            i_mask = indices[self.ground_truth == i]
            self.gt_flipped[i] = i_mask
            self.labeled = self.labeled.union(set(self.fid[i]))
        self.unlabeled = sorted(list(set(indices) - self.labeled))
        self.labeled = sorted(list(self.labeled))

        self.have_useful = True


    def plot_initial(self):
        if self.X.shape[1] > 2:
            print('Sorry, plotting for higher dimensional datasets is not yet implemented..')
            return
        if not self.have_useful:
            self.get_useful_structs()
        if min(self.classes) == -1:
            _, stats = calc_stats_multi(self.ground_truth, self.fid, self.gt_flipped)
            plot_iter(stats, self.X, k_next=-1)
        else:
            indices = np.array(list(range(self.N)))  # 0, 1, 2, ... for use in other calculations
            m_dummy = np.zeros((self.N, self.num_class))
            for i in range(self.num_class):
                i_mask = indices[self.ground_truth == self.classes[i]]
                m_dummy[i_mask, i] = 1.
            _, stats = calc_stats_multi(m_dummy, self.fid, self.gt_flipped)
            plot_iter_multi(stats, self.X, self.fid, k_next=-1)


    def plot_iteration(self, u):
        if not self.have_useful:
            self.get_useful_structs()
        if min(self.classes) == -1:
            _, stats = calc_stats_multi(self.ground_truth, self.fid, self.gt_flipped)
            plot_iter(stats, self.X, k_next=-1)
        else:
            indices = np.array(list(range(self.N)))  # 0, 1, 2, ... for use in other calculations
            m_dummy = np.zeros((self.N, self.num_class))
            for i in range(self.num_class):
                i_mask = indices[self.ground_truth == self.classes[i]]
                m_dummy[i_mask, i] = 1.
            _, stats = calc_stats_multi(m_dummy, self.fid, self.gt_flipped)
            plot_iter_multi(stats, self.X, self.fid, k_next=-1)


    ### function to reset Data object to orig labeled, fid, unlabeled




def load_2_moons(N=2000, noise=0.2, sup_percent=0.05, normed_lap=False, random=False, zero_one=False):
    # rand_state = None yields a random, new dataset to be made
    if random:
        rand_state = None
    else:
        rand_state = 10

    # call the sklearn function for making moons data
    X, ground_truth = make_moons(n_samples=N, noise=noise, random_state=rand_state)

    # if want +1, -1 classes, change the corresponding entries
    if not zero_one:
        ground_truth[np.where(ground_truth == 0)] = -1

    # Define the fidelity dictionary
    classes = np.unique(ground_truth)
    indices = np.array(list(range(N)))
    fid = {}
    for i in classes:
        i_mask = indices[ground_truth ==i]
        np.random.shuffle(i_mask)
        n_i = len(i_mask)
        fid[i] = list(i_mask[:int(sup_percent*n_i)])

    # Graph Generation and Eigen-Calculation
    W = make_sim_graph(X)
    evals, evecs = get_eig_Lnorm(W, normed_=normed_lap)

    return Data_obj(X, evals, evecs, fid, ground_truth)


def load_gaussian_cluster(Ns, means, covs, sup_percent=0.05, normed_lap=False):
    if len(Ns) != len(means):
        raise ValueError('Must have same number of means as clusters in Ns')
    if len(covs) != len(means):
        raise ValueError('Must have same number of means as covariance matrices')
    N = sum(Ns)
    classes = [i for i in range(len(Ns))]
    X, W, ground_truth = generate_data_graphs(Ns, means, covs)
    evals, evecs = get_eig_Lnorm(W, normed_=normed_lap)

    indices = np.array(list(range(N)))
    fid = {}
    for i in classes:
        i_mask = indices[ground_truth ==i]
        np.random.shuffle(i_mask)
        n_i = len(i_mask)
        fid[i] = list(i_mask[:int(sup_percent*n_i)])

    return Data_obj(X, evals, evecs, fid, ground_truth)





################# Graph generation and other Calculations

def sqdist(X, Y):
    # Michael Luo code  - X is (d x m) np array, Y is (d x n) np array
    # returns D_ij = |x_i - y_j|^2 matrix
    # better than KD tree for larger dimensions?
    m = X.shape[1]
    n = Y.shape[1]
    Yt = Y.T
    XX = np.sum(X*X, axis=0)
    YY = np.sum(Yt*Yt, axis=1).reshape(n,1)
    return np.tile(XX, (n,1)) + np.tile(YY, (1,m)) - 2*Yt.dot(X)

# Make the similarity graph from rows of X
def make_sim_graph(X, k_nn=5):
    N = X.shape[0]
    # Make weighted similarity graph, in W
    D = sqdist(X.T,X.T)
    ind_knn = np.argsort(D, axis=1)[:,1:k_nn+1]
    Dknn = D[(np.arange(N).reshape(N,1),ind_knn)]

    I = np.tile(np.arange(N).reshape(N,1), (1,k_nn)).flatten()
    J = ind_knn.flatten()
    Dmean = (np.sum(Dknn, axis=1)/k_nn).reshape(N,1)
    w_sp = np.divide(Dknn, Dmean)
    w_sp = np.exp(-(w_sp * w_sp))
    W = sps.csr_matrix((w_sp.flatten() , (I, J)), shape=(N,N))
    W = 0.5*(W+W.T)

    return W


# Gaussian clusters data
def generate_data_graphs(Ns, means, Covs, k_nn=5):
    '''
    inputs   :
      Ns    : class sizes, an array of length K
      means : class means, list of mean vectors in R^d
      Covs  : class covariances, list of Cov matrices of size d x d
    returns  : adjacency matrix
    '''
    N = sum(Ns)
    d = len(means[0])
    X = np.zeros((N, d))
    ground_truth = np.zeros(N)
    offset = 0
    for i in range(len(Ns)):
        Ni = Ns[i]
        X[offset:offset+Ni,:] = np.random.multivariate_normal(means[i], Covs[i], Ni)
        ground_truth[offset: offset+Ni] = i
        offset += Ni


    # Make weighted similarity graph, in W
    W = make_sim_graph(X, k_nn)
    return X, W, ground_truth


def get_eig_Lnorm(W, return_L=False, normed_=True):
    L_sym = sps.csgraph.laplacian(W, normed=normed_)
    [w, v] = sla.eigh(L_sym.toarray())
    if return_L:
        return w, v, L_sym.toarray()
    return w, v
