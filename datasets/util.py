import numpy as np
import scipy.sparse as sps
import scipy.linalg as sla
import matplotlib.pyplot as plt


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

# Making a similarity graph
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
    offset = 0
    for i in range(len(Ns)):
        Ni = Ns[i]
        X[offset:offset+Ni,:] = np.random.multivariate_normal(means[i], Covs[i], Ni)
        offset += Ni


    # Make weighted similarity graph, in W
    W = make_sim_graph(X, k_nn)
    return X, W


def get_eig_Lnorm(W, return_L=False, normed_=True):
    L_sym = sps.csgraph.laplacian(W, normed=normed_)
    [w, v] = sla.eigh(L_sym.toarray())
    if return_L:
        return w, v, L_sym.toarray()
    return w, v



def threshold1D(u):
    u[u >= 0.0] = 1.
    u[u < 0.0] = -1.
    return u

def threshold2D(u):
    thresh_ind = np.argmax(u, axis=1)
    u *= 0.
    u[np.ix_([i for i in range(u.shape[0])], thresh_ind)] = 1.
    return u

def threshold2D_many(u_samples):
    print('Not working... will cause error')
    thresh_ind = np.argmax(u_samples, axis=1)
    u_samples[:,:,:] = 0
    first_ind = np.array([[i for i in range(u_samples.shape[0])] for j in range(u_samples.shape[2])])
    third_ind = np.array([[i for j in range(u_samples.shape[2])] for i in range(u_samples.shape[0])])
    print(first_ind)
    print(first_ind.flatten())
    print(third_ind)
    print(third_ind.flatten())
    u_samples[np.ix_(first_ind.flatten(), thresh_ind.flatten(), third_ind.flatten())] = 1.
    return u_samples

def threshold1D_avg(u_samples):
    return np.average(1.*(u_samples >= 0.), axis=1) # ith entry is estimated mean that ith node is of class 1

def threshold2D_avg(u_samples):
    print('deprecated... need to fix')
    N, num_class, n_samples = u_samples.shape
    thresh_ind = np.argmax(u_samples, axis=1)
    u_thresh_avg = np.zeros((N, num_class))
    for i in range(num_class):
        u_thresh_avg[:,i] = np.bincount(thresh_ind, axis=1)/N
    return u_thresh_avg



def plot_iter(stats, X, k_next=-1):
    corr1 = stats['corr1']
    corr2 = stats['corr2']
    sup1 = stats['sup1']
    sup2 = stats['sup2']
    incorr1 = stats['incorr1']
    incorr2 = stats['incorr2']
    if type(k_next) == type([1, 2]):
        plt.scatter(X[np.ix_(k_next,[0])], X[np.ix_(k_next,[1])], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title('Dataset with Label for %s added' % str(k_next))
    elif k_next >= 0:
        plt.scatter(X[k_next,0], X[k_next,1], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title('Dataset with Label for %s added' % str(k_next))
    elif k_next == -1:
        plt.title('Dataset with Initial Labeling')

    plt.scatter(X[corr1,0], X[corr1,1], marker='x', c='b', alpha=0.2)
    plt.scatter(X[incorr1,0], X[incorr1,1], marker='x', c='r', alpha=0.2)
    plt.scatter(X[corr2,0], X[corr2,1], marker='o', c='r',alpha=0.15)
    plt.scatter(X[incorr2,0], X[incorr2,1], marker='o', c='b',alpha=0.15)
    plt.scatter(X[sup1,0], X[sup1,1], marker='x', c='b', alpha=1.0)
    plt.scatter(X[sup2,0], X[sup2,1], marker='o', c='r', alpha=1.0)
    plt.axis('equal')
    plt.show()
    return




def calc_stats_multi(m, fid, gt_flipped, _print=False):
    stats = {}

    N = m.shape[0]
    classes = list(fid.keys())
    offset = min(classes)

    if offset == -1:
        m1 = np.where(m >= 0)[0]
        m2 = np.where(m < 0)[0]


        sup1 = fid[1]
        sup2 = fid[-1]
        corr1 = list(set(m1).intersection(set(gt_flipped[1])))
        incorr1 = list(set(m2).intersection(set(gt_flipped[1])))
        corr2 = list(set(m2).intersection(set(gt_flipped[-1])))
        incorr2 = list(set(m1).intersection(set(gt_flipped[-1])))

        stats['corr1'] = corr1
        stats['corr2'] = corr2
        stats['sup1'] = sup1
        stats['sup2'] = sup2
        stats['incorr1'] = incorr1
        stats['incorr2'] = incorr2

        error = ((len(incorr1) + len(incorr2))/N )

    else:
        m_class = np.argmax(m,axis=1) + offset # get the class labelings for the current m
        m_class_ind = {c:np.where(m_class == c)[0] for c in classes}

        corr = {c : list(set(m_class_ind[c]).intersection(set(gt_flipped[c]))) for c in classes}
        incorr = {}
        for corr_class, wr_class in permutations(classes, 2):
            if corr_class not in incorr.keys():
                incorr[corr_class] = [(wr_class, list(set(m_class_ind[wr_class]).intersection(set(gt_flipped[corr_class]))))]
            else:
                incorr[corr_class].append((wr_class, list(set(m_class_ind[wr_class]).intersection(set(gt_flipped[corr_class])))))

        stats['corr'] = corr
        stats['incorr'] = incorr

        tot_correct = 0.
        for corr_c_nodes in corr.values():
            tot_correct += len(corr_c_nodes)
        acc = tot_correct/N
        error = 1. - acc


    if _print:
        print('Error = %f' % error )

    return error, stats


COLORS = ['b', 'r', 'g',  'cyan', 'k', 'y']
MARKERS = ['x', 'o', '^', '+', 'v']

def plot_iter_multi(stats, X, fid, k_next=-1):
    corr = stats['corr']
    incorr = stats['incorr']
    if type(k_next) == type([1, 2]):
        plt.scatter(X[np.ix_(k_next,[0])], X[np.ix_(k_next,[1])], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title('Dataset with Label for %s added' % str(k_next))
    elif k_next >= 0:
        plt.scatter(X[k_next,0], X[k_next,1], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title('Dataset with Label for %s added' % str(k_next))
    elif k_next == -1:
        plt.title('Dataset with Initial Labeling')

    classes = corr.keys()
    ofs = min(classes)
    num_class = len(classes)

    if num_class > len(COLORS):
        print('Not enough colors in COLORS list, reusing colors with other markers.')


    for c, c_nodes in corr.items():
        plt.scatter(X[c_nodes,0], X[c_nodes,1], marker=MARKERS[(c-ofs)%num_class], c=COLORS[(c-ofs)%num_class], alpha=0.2)
    for corr_c, vals in incorr.items():
        incorr_c, incorr_nodes = zip(*vals)
        for l in range(len(incorr_c)):
            plt.scatter(X[incorr_nodes[l],0], X[incorr_nodes[l],1], marker=MARKERS[(corr_c-ofs)%num_class],
                                        c=COLORS[(incorr_c[l]-ofs)%num_class], alpha=0.2)
    for c, fid_c in fid.items():
        plt.scatter(X[fid_c,0], X[fid_c,1], marker=MARKERS[(c-ofs)%num_class], c=COLORS[(c-ofs)%num_class], alpha=1.0)
    plt.axis('equal')
    plt.show()
    return




def calc_orig_multi(v, w, fid, labeled, unlabeled, tau, alpha, gamma2):
    N = v.shape[0]
    classes = fid.keys()
    num_class = len(classes)
    ofs = min(classes)

    if -1 in fid.keys():
        y = np.zeros(N)  # this will already be in the expanded size, as if (H^Ty)
        y[fid[1]] = 1.
        y[fid[-1]] = -1.
    else:
        y = np.zeros((N, num_class))  # this will already be in the expanded size, as if (H^Ty)
        for c, fid_c in fid.items():
            y[fid_c,c-ofs] = 1    # TODO: make it sparse

    N_prime = len(labeled)
    #w_inv = (tau ** (2 * alpha)) * np.power(w + tau**2., -alpha)     # diagonalization of C_t,e
    w_inv =  np.power(w + tau**2., -alpha)     # diagonalization of C_t,e
    C_tau = v.dot((v*w_inv).T)
    C_ll = C_tau[np.ix_(labeled, labeled)]
    C_all_l = C_tau[:,labeled]
    C_ll[np.diag_indices(N_prime)] += gamma2  # directly changing C_ll
    A_inv = sla.inv(C_ll)
    Block1 = C_all_l.dot(A_inv)
    C = C_tau - Block1.dot(C_all_l.T)
    m = Block1.dot(y[labeled])
    if -1 in fid.keys():
        m = np.asarray(m).flatten()
    else:
        m = np.asarray(m)
    return m, np.asarray(C), y
