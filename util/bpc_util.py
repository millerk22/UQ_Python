import numpy as np
import scipy as sp
from scipy.sparse import csgraph, csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import lsq_linear
import scipy.linalg as sla
import matplotlib.pyplot as plt
from util.plotutil import *
from itertools import product
from scipy.optimize import newton, root_scalar

""""""""""""""""""""""""""""" SYNTHETIC TEST FUNCTIONS """""""""""""""""""""

def get_W0(Ns, seed=None, density=1.0):
    '''
    inputs   :
      Ns    : class sizes, an array of length K
      seed  : random seed
    returns  : adjacency matrix
    '''
    W0s = []
    for Ni in Ns:
        W0s.append(sp.sparse.random(Ni, Ni, density=density, random_state=seed))
    W0 = sp.sparse.block_diag(W0s, format='csr')
    W0 += W0.T
    W0 /= 2
    # remove diagonals
    W0 -= sp.sparse.diags(W0.diagonal())
    return W0


def generate_random_graphs_rest(Ns, eps, W_0, perturb_level, seed):
    '''
    inputs   :
      Ns    : class sizes, an array of length K
      eps   : epsilon
      W_0   : the separated weight matrix, will be common among all iterations
      perturb_level  : W = W_0 + eps W_1 + ... + eps^level W_level
      seed  : random seed
    returns  : adjacency matrix
    '''
    density = 1.0
    W = W_0.copy().toarray() # convert sparse W_0 to dense for W
    nnz = np.nonzero(W_0)
    shape = W.shape
    for i in range(1,perturb_level+1):
        Wh = np.random.rand(*shape)
        Wh[nnz] = 0.
        W += (eps**i) * Wh
    # make sure W is symmetric
    W += W.T
    W /= 2
    # remove diagonals
    np.fill_diagonal(W, 0)

    return W



def syn_overall_setup(rand=False,Ns=[100,100,100],num_in_each=5, normalized=True, density_=1.0, seed=10):
    """ Get overall setup for synthetic dataset, with ground truth u being in the
    span of the properly normalized Chi vectors, according to the degree of nodes.

    Inputs:
        rand : bool, if True, then ground truth u will be random linear combination of Chis
        Ns : list of number of points in each class/cluster
        num_in_each : # of fidelity points in each cluster.
        normalized : bool, whether or not we are using the normalized graph Laplacian or not.
                        default is True.

    Outputs:
        Ns : return the list of Ns
        u : the ground truth function we will be comparing against
        B : the supervision matrix, sparse diagonal matrix with ii th entry = 1 if labeled.
        labeled : indices of the labeled nodes
        draws : random numbers for use in defining observations function y
        W_0 : the underlying separated graph that we will be adding epsilon perturbations onto.
    """

    N = sum(Ns)

    W_0 = get_W0(Ns, seed=seed, density=density_)
    if normalized:
        D_0 = np.asarray(np.sum(W_0, axis=1)).flatten()**0.5   # normalization for the


    # begin with ground truth, u
    u = np.zeros((N,len(Ns)))
    fid = np.zeros(N, dtype='int')

    # The eigenvectors of the W_0 matrix
    Chis = np.zeros((N,len(Ns)))
    idx = 0
    for l in range(len(Ns)):
        fid[idx:idx+num_in_each] = 1
        Chis[idx:idx+Ns[l],l] = 1./np.sqrt(Ns[l])
        if normalized:
            Chis[idx:idx+Ns[l],l] *= D_0[idx:idx+Ns[l]]
            Chis[idx:idx+Ns[l],l] /= sp.linalg.norm(Chis[idx:idx+Ns[l],l])
        idx += Ns[l]

    if rand: # each column is a random linear combination of columns of Chis
        u = Chis.dot(np.random.randn(len(Ns), len(Ns)))

    else:     # each column is a random scaling of corresponding columns of Chis
        u = Chis.dot(np.diag(np.random.randn(len(Ns))))

    B = sp.sparse.diags(fid, format='csr')
    labeled = np.nonzero(fid)

    draws = np.random.randn(len(Ns),len(labeled[0]))  # get draws for the observations y, dependent on gamma later on

    return Ns, u, B, labeled, draws, W_0







""" Eigen calculations"""

'''




def get_eig_Lnorm(eps, Ns):
    # Calculate the matrices to be involved. ensure we draw the same graph each time for consistency
    W = generate_random_graphs(Ns, eps, perturb_level=3, seed=10)
    L_sym = csgraph.laplacian(W, normed=True)
    [w, v] = sp.linalg.eigh(L_sym.toarray())

    return w, v


def get_eig(eps, Ns):
    # Calculate the matrices to be involved. ensure we draw the same graph each time for consistency
    W = generate_random_graphs(Ns, eps, perturb_level=3, seed=10)
    L = csgraph.laplacian(W, normed=False)
    [w, v] = sp.linalg.eigh(L.toarray())

    return w, v
'''



def get_eig_Lnorm_withW0(eps, Ns, W_0, normalized=True, seed=None):
    # Calculate the matrices to be involved. ensure we draw the same graph each time for consistency
    W = generate_random_graphs_rest(Ns, eps, W_0, perturb_level=3, seed=seed)
    L = csgraph.laplacian(W, normed=normalized)
    #print("Graph is normalized Laplacian = %s" % str(normalized))
    try:
        w, v = sp.linalg.eigh(L.toarray())
    except:
        w, v = sp.linalg.eigh(L)
    return w, v





""" Tests Code"""


def syn_run_test(T, ALPHAS, Ns, B, labeled, u, W_0, normalized=True, little_oh=False, all_norms=False, seed=None):
    """
    Calculate a LOT of different values, now with the input being normalized.
    """
    TRC = np.zeros((len(ALPHAS), len(T)))
    TRCBC = np.zeros((len(ALPHAS), len(T)))
    BIAS = np.zeros((len(ALPHAS), len(T)))

    if all_norms:
        normC = np.zeros((len(ALPHAS), len(T)))
        normC_tau_inv_u = np.zeros((len(ALPHAS), len(T)))
        normC_tau_inv_u_p1 = np.zeros((len(ALPHAS), len(T)))
        normC_tau_inv_u_p2 = np.zeros((len(ALPHAS), len(T)))


    K = len(Ns)
    N = sum(Ns)
    u_n = u[:,0]/sp.linalg.norm(u[:,0])   # normalized vector for use in the BIAS calculation

    # For each value of tau given in the list T
    for j, tau in enumerate(T):
        if little_oh:
            eps = tau**3.
        else:
            eps = tau**2.

        # Calculate the eigenvalues/vecs for this level of epsilon and tau value
        w, v = get_eig_Lnorm_withW0(eps, Ns, W_0, normalized, seed)

        for i,alpha in enumerate(ALPHAS):
            gamma = tau**alpha
            d_inv = (tau ** (-2. * alpha)) * np.power(w + tau**2., alpha)     # diagonalization of C_t,e - will be used later


            if all_norms: # only calculate the parts of the triangle inequality if want to track all norms
                C_tau_inv_u = (v*d_inv.reshape(1,N)).dot(v.T.dot(u_n))
                normC_tau_inv_u[i,j] = sp.linalg.norm(C_tau_inv_u)

                # first term of the triangle inequality
                vK = v[:,:K]                 # using only the first K eigenvectors of C_\tau, \eps^{-1}
                dK = d_inv[:K]
                C_tau_inv_u_p1 = (vK*dK.reshape(1,K)).dot(vK.T.dot(u_n))
                normC_tau_inv_u_p1[i,j] = sp.linalg.norm(C_tau_inv_u_p1)

                # second term of the triangle inequality, using the K+1,...,Nth eigenpairs of C_\tau,\eps^{-1}
                v_K = v[:,K:]
                d_K = d_inv[K:]
                C_tau_inv_u_p2 = (v_K*d_K.reshape(1,N-K)).dot(v_K.T.dot(u_n))
                normC_tau_inv_u_p2[i,j] = sp.linalg.norm(C_tau_inv_u_p2)

            # prior_inv : C_{tau,eps}^{-1}, where
            # C_{tau, eps}^{-1} = tau^{-2alpha}(L + tau^2 I)^alpha
            prior_inv = v.dot(sp.sparse.diags(d_inv, format='csr').dot(v.T))

            # B/gamma^2
            B_over_gamma2 = B / (gamma * gamma)
            # post_inv  : (B/gamma^2 + C_{tau,\eps}^{-1})^{-1}
            post_inv  = prior_inv + B_over_gamma2
            # C^{-1}
            post = post_inv.I
            bias = sp.linalg.norm(post.dot(prior_inv.dot(u_n)))
            BIAS[i,j] = bias**2.
            if all_norms:
                normC[i,j] = sp.linalg.norm(post)

            # Calculate Tr(C)
            trC = sp.trace(post)
            TRC[i,j] = trC

            # Calculate Tr(CBC)/gamma^2
            post2 = post.dot(post)
            trCBC = sp.trace(post2[np.ix_(labeled[0], labeled[0])])
            TRCBC[i,j] = trCBC/(gamma**2.)

    if all_norms:
        return TRC, TRCBC, normC, normC_tau_inv_u, normC_tau_inv_u_p1, normC_tau_inv_u_p2, BIAS

    return TRC, TRCBC, BIAS


def syn_run_test_new(T, ALPHAS, Ns, B, labeled, u, W_0, normalized=True, beta=-1, all_norms=False, seed=None):
    """
    Calculate a LOT of different values, now with the input being normalized.
    """
    TRC = np.zeros((len(ALPHAS), len(T)))
    TRCBC = np.zeros((len(ALPHAS), len(T)))
    BIAS = np.zeros((len(ALPHAS), len(T)))

    u_n = u[:,0]/sp.linalg.norm(u[:,0])   # normalized vector for use in the BIAS calculation

    if beta < 0:
        w, v = get_eig_Lnorm(W_0, normed_=normalized)
    # For each value of tau given in the list T
    for j, tau in enumerate(T):
        if beta >= 2.0:
            eps = tau**beta
            # Calculate the eigenvalues/vecs for this level of epsilon and tau value
            w, v = get_eig_Lnorm_withW0(eps, Ns, W_0, normalized, seed)

        for i,alpha in enumerate(ALPHAS):
            gamma = tau**alpha
            d_inv = (tau ** (-2. * alpha)) * np.power(w + tau**2., alpha)     # diagonalization of C_t,e - will be used later

            # prior_inv : C_{tau,eps}^{-1}, where
            # C_{tau, eps}^{-1} = tau^{-2alpha}(L + tau^2 I)^alpha
            prior_inv = v.dot(sp.sparse.diags(d_inv, format='csr').dot(v.T))

            # B/gamma^2
            B_over_gamma2 = B / (gamma * gamma)
            # post_inv  : (B/gamma^2 + C_{tau,\eps}^{-1})^{-1}
            post_inv  = prior_inv + B_over_gamma2
            # C^{-1}
            post = post_inv.I
            bias = sp.linalg.norm(post.dot(prior_inv.dot(u_n)))
            BIAS[i,j] = bias**2.

            # Calculate Tr(C)
            trC = sp.trace(post)
            TRC[i,j] = trC

            # Calculate Tr(CBC)/gamma^2
            post2 = post.dot(post)
            trCBC = sp.trace(post2[np.ix_(labeled[0], labeled[0])])
            TRCBC[i,j] = trCBC/(gamma**2.)

    return TRC, TRCBC, BIAS


def syn_run_test0(T, ALPHAS, Ns, B, labeled, u, W_0, normalized=True, seed=None):
    """
    Calculate a LOT of different values, now with the input being normalized.
    """
    TRC = np.zeros((len(ALPHAS), len(T)))
    TRCBC = np.zeros((len(ALPHAS), len(T)))
    BIAS = np.zeros((len(ALPHAS), len(T)))

    K = len(Ns)
    N = sum(Ns)
    u_n = u[:,0]/sp.linalg.norm(u[:,0])   # normalized vector for use in the BIAS calculation

    w, v = get_eig_Lnorm(W_0, normed_=normalized)


    # For each value of tau given in the list T
    for j, tau in enumerate(T):


        for i,alpha in enumerate(ALPHAS):
            d_inv = (tau ** (-2. * alpha)) * np.power(w + tau**2., alpha)     # diagonalization of C_t,0 - will be used later

            # prior_inv : C_{tau,0}^{-1}, where
            # C_{tau, 0}^{-1} = tau^{-2alpha}(L + tau^2 I)^alpha
            prior_inv = v.dot(sp.sparse.diags(d_inv, format='csr').dot(v.T))

            gamma = tau**alpha

            # B/gamma^2
            B_over_gamma2 = B / (gamma * gamma)
            # post_inv  : (B/gamma^2 + C_{tau,\eps}^{-1})^{-1}
            post_inv  = prior_inv + B_over_gamma2
            # C^{-1}
            post = post_inv.I
            bias = sp.linalg.norm(post.dot(prior_inv.dot(u_n)))
            BIAS[i,j] = bias**2.

            # Calculate Tr(C)
            trC = sp.trace(post)
            TRC[i,j] = trC

            # Calculate Tr(CBC)/gamma^2
            post2 = post.dot(post)
            trCBC = sp.trace(post2[np.ix_(labeled[0], labeled[0])])
            TRCBC[i,j] = trCBC/(gamma**2.)

    return TRC, TRCBC, BIAS


def syn_run_test_gamma(G, ALPHAS, Ns, B, labeled, u, W_0, normalized=True, beta=-1, seed=None):
    """
    Calculate a LOT of different values, now with the input being normalized.
    """
    TRC = np.zeros((len(ALPHAS), len(G)))
    TRCBC = np.zeros((len(ALPHAS), len(G)))
    BIAS = np.zeros((len(ALPHAS), len(G)))

    u_n = u[:,0]/sp.linalg.norm(u[:,0])   # normalized vector for use in the BIAS calculation

    if beta < 0:
        w, v = get_eig_Lnorm(W_0, normed_=normalized)

    # For each value of tau given in the list T
    for i,alpha in enumerate(ALPHAS):
        for j, gamma in enumerate(G):

            tau = gamma**(1./alpha)# Calculate the eigenvalues/vecs for this level of epsilon and tau value
            print(alpha, gamma, tau)
            if beta > 0:
                eps = tau**beta
                w, v = get_eig_Lnorm_withW0(eps, Ns, W_0, normalized, seed)


            d_inv = (tau ** (-2. * alpha)) * np.power(w + tau**2., alpha)     # diagonalization of C_t,e - will be used later

            # prior_inv : C_{tau,eps}^{-1}, where
            # C_{tau, eps}^{-1} = tau^{-2alpha}(L + tau^2 I)^alpha
            prior_inv = v.dot(sp.sparse.diags(d_inv, format='csr').dot(v.T))

            # B/gamma^2
            B_over_gamma2 = B / (gamma * gamma)
            # post_inv  : (B/gamma^2 + C_{tau,\eps}^{-1})^{-1}
            post_inv  = prior_inv + B_over_gamma2
            # C^{-1}
            post = post_inv.I
            bias = sp.linalg.norm(post.dot(prior_inv.dot(u_n)))
            BIAS[i,j] = bias**2.

            # Calculate Tr(C)
            trC = sp.trace(post)
            TRC[i,j] = trC

            # Calculate Tr(CBC)/gamma^2
            post2 = post.dot(post)
            trCBC = sp.trace(post2[np.ix_(labeled[0], labeled[0])])
            TRCBC[i,j] = trCBC/(gamma**2.)

    return TRC, TRCBC, BIAS


def syn_run_test0_gamma(T, ALPHAS, Ns, B, labeled, u, W_0, normalized=True, seed=None):
    """
    Calculate a LOT of different values, now with the input being normalized.
    """
    TRC = np.zeros((len(ALPHAS), len(T)))
    TRCBC = np.zeros((len(ALPHAS), len(T)))
    BIAS = np.zeros((len(ALPHAS), len(T)))

    K = len(Ns)
    N = sum(Ns)
    u_n = u[:,0]/sp.linalg.norm(u[:,0])   # normalized vector for use in the BIAS calculation

    w, v = get_eig_Lnorm(W_0, normed_=normalized)


    # For each value of tau given in the list T
    for j, tau in enumerate(T):
        for i,alpha in enumerate(ALPHAS):
            d_inv = (tau ** (-2. * alpha)) * np.power(w + tau**2., alpha)     # diagonalization of C_t,0 - will be used later

            # prior_inv : C_{tau,0}^{-1}, where
            # C_{tau, 0}^{-1} = tau^{-2alpha}(L + tau^2 I)^alpha
            prior_inv = v.dot(sp.sparse.diags(d_inv, format='csr').dot(v.T))

            gamma = tau**alpha

            # B/gamma^2
            B_over_gamma2 = B / (gamma * gamma)
            # post_inv  : (B/gamma^2 + C_{tau,\eps}^{-1})^{-1}
            post_inv  = prior_inv + B_over_gamma2
            # C^{-1}
            post = post_inv.I
            bias = sp.linalg.norm(post.dot(prior_inv.dot(u_n)))
            BIAS[i,j] = bias**2.

            # Calculate Tr(C)
            trC = sp.trace(post)
            TRC[i,j] = trC

            # Calculate Tr(CBC)/gamma^2
            post2 = post.dot(post)
            trCBC = sp.trace(post2[np.ix_(labeled[0], labeled[0])])
            TRCBC[i,j] = trCBC/(gamma**2.)

    return TRC, TRCBC, BIAS







""" Plotting Code """

def syn_plot_data(T, data, ALPHAS, param_str, title_=r'$\mathrm{Tr}(C^*)$', val_str="TRC", save=False, Jval=-1, _fontsize=25, little_oh=False):
    """
        Generic plotting function for showing values contained in the matrix ``data``
        and title given in the input string ``title_``.

        If want to fit the the line, input value Jval >= 0. (the index where to start)
    """
    n_alpha = len(ALPHAS)
    if Jval >= 0:
        # Fit the line for T[J[i]:], TRC[i,:]
        print('Line fitting for %s' % val_str)
        J = n_alpha*[Jval]
        line_stats = np.zeros((n_alpha,2))
        for i in range(n_alpha):
            j = J[i]
            t = np.log(T[j:])
            A = np.array([t, np.ones(len(t))]).T
            b = np.log(data[i,j:])
            res = lsq_linear(A, b)
            line_stats[i,:] = res.x
            print('The slope for alpha = %1.1f is : %2.4f' % (ALPHAS[i], res.x[0]))

    markers = ['o','*', 'v', '^', '>', 's', '<', 'p', 'h', 'o','*', 'v', '^', '>', 's', '<', 'p' ]
    fig = plt.figure()
    ax = fig.gca()
    ax.loglog(T, data.T)
    for i, alpha in enumerate(ALPHAS):
        ax.scatter(T, data[i,:], marker=markers[i], label=r'$\alpha =$ %2.1f'% alpha)
    plt.xlabel(r'$\tau$', fontsize=_fontsize)
    plt.ylabel(title_, fontsize=_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=_fontsize)
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        print("Saving figure at ./figures/paper/%s_%s.png" % (param_str, val_str))
        plt.savefig('./figures/BPCpaper/%s_%s.png' % (param_str, val_str))
    else:
        if little_oh:
            plt.title(title_ + r', $\epsilon = \tau^3, \gamma = \tau^\alpha$', fontsize=15)
        else:
            plt.title(title_ + r', $\epsilon = \tau^2, \gamma = \tau^\alpha$', fontsize=15)
    plt.show()

    return


def syn_plot_data_gamma(T, data, ALPHAS, param_str, file2save=None, t_ranges=None, plot_ranges=None,title_=r'$\mathrm{Tr}(C^*)$', val_str="TRC", save=False, Jval=-1, _fontsize=25, beta=-1, legend=False):
    """
        Generic plotting function for showing values contained in the matrix ``data``
        and title given in the input string ``title_``.

        If want to fit the the line, input value Jval >= 0. (the index where to start)
    """
    file_string = param_str + '_'+ val_str+ '\nalpha,slope (in terms of gamma exponent)\n'
    n_tau = len(T)
    n_alpha = len(ALPHAS)
    if t_ranges is None:
        t_ranges = [(0,n_tau) for i in range(n_alpha)]
    if plot_ranges is None:
        plot_ranges = n_alpha*[(0,n_tau)]


    # Fit the line for T[J[i]:], data[i,:]
    print('Line fitting for %s' % val_str)
    line_stats = np.zeros((n_alpha,2))
    for i in range(n_alpha):
        j_start, j_end = t_ranges[i]
        t = np.log(T[j_start:j_end])
        A = np.array([t, np.ones(len(t))]).T
        b = np.log(data[i,j_start:j_end])
        res = lsq_linear(A, b)
        line_stats[i,:] = res.x
        print('The slope (in terms of gamma) for alpha = %1.1f is : %2.4f' % (ALPHAS[i], res.x[0]/ALPHAS[i]))
        #print('The slope (in terms of gamma) for alpha = %1.1f is : %2.4f' % (ALPHAS[i], res.x[0]))
        file_string += '%1.1f, %2.4f\n' % (ALPHAS[i], res.x[0]/ALPHAS[i])

    markers = ['o','*', 'v', '^', '>', 's', '<', 'p', 'h', 'o','*', 'v', '^', '>', 's', '<', 'p' ]
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors += prop_cycle.by_key()['color']

    fig = plt.figure()
    ax = fig.gca()

    for i, alpha in enumerate(ALPHAS):
        j_start, j_end = plot_ranges[i]
        ax.loglog(T[j_start:j_end], data[i,j_start:j_end], c=colors[i])
        ax.scatter(T[j_start:j_end], data[i,j_start:j_end], marker=markers[i], c=colors[i], label=r'$\alpha =$ %2.1f'% alpha)
    plt.xlabel(r'$\tau=\gamma^{1/\alpha}$', fontsize=_fontsize)
    plt.ylabel(title_, fontsize=_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=_fontsize)
    if legend:
        plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        if save:
            if file2save is None:
                print("Saving figure at ./figures/BPCpaper/%s_%s.png" % (param_str, val_str))
                plt.savefig('./figures/BPCpaper/%s_%s.png' % (param_str, val_str))
                with open('./figures/BPCpaper/%s_%s.txt' % (param_str, val_str), 'w') as file:
                    file.write(file_string)
            else:
                print("Saving figure at %s%s_%s.png" % (file2save, param_str, val_str))
                plt.savefig('%s%s_%s.png' % (file2save, param_str, val_str))
                with open('%s%s_%s.txt' % (file2save, param_str, val_str), 'w') as file:
                    file.write(file_string)
    else:
        if beta < 0:
            plt.title(title_ + r', $\epsilon = 0, \tau = \gamma^\alpha$', fontsize=15)
        else:
            plt.title(title_ + r', $\epsilon = \gamma^{%d/\alpha}, \tau = \gamma^\alpha$' % beta, fontsize=15)
    plt.show()

    return


def syn_plot_data_range(T, data, ALPHAS, param_str, file2save=None, t_ranges=None,title_=r'$\mathrm{Tr}(C^*)$', val_str="TRC", save=False, _loc='lower right',_fontsize=25, _rms=150, little_oh=False, zero=False):
    """
        Generic plotting function for showing values contained in the matrix ``data``
        and title given in the input string ``title_``.
    """
    file_string = param_str + '_'+ val_str+ '\nalpha,slope\n'
    n_alpha = len(ALPHAS)
    n_tau = len(T)
    if t_ranges is None:
        t_ranges = [(0,n_tau) for i in range(n_alpha)]
    # Fit the line for T[J[i]:], data[i,:]
    print('Line fitting for %s' % val_str)
    line_stats = np.zeros((n_alpha,2))
    for i in range(n_alpha):
        j_start, j_end = t_ranges[i]
        t = np.log(T[j_start:j_end])
        A = np.array([t, np.ones(len(t))]).T
        b = np.log(data[i,j_start:j_end])
        res = lsq_linear(A, b)
        line_stats[i,:] = res.x
        print('The slope for alpha = %1.1f is : %2.4f' % (ALPHAS[i], res.x[0]))
        file_string += '%1.1f, %2.4f\n' % (ALPHAS[i], res.x[0])


    markers = ['o','*', 'v', '^', '>', 's', '<', 'p', 'h', 'o','*', 'v', '^', '>', 's', '<', 'p' ]
    #colors = ['b', 'r', 'g', 'purple', 'cyan', 'k','orange' , 'brown']
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors += prop_cycle.by_key()['color']

    fig = plt.figure()
    ax = fig.gca()
    #ax.loglog(T, data.T)
    for i, alpha in enumerate(ALPHAS):
        ax.loglog(T, data[i,:], c=colors[i])
        ax.scatter(T, data[i,:], marker=markers[i], c=colors[i], label=r'$\alpha =$ %2.1f'% alpha)
        ax.scatter(T[t_ranges[i][0]],data[i,t_ranges[i][0]], c=colors[i], marker=markers[i], s=_rms)
        ax.scatter(T[t_ranges[i][1]-1],data[i,t_ranges[i][1]-1], c=colors[i], marker=markers[i], s=_rms)
    plt.xlabel(r'$\tau$', fontsize=_fontsize)
    plt.ylabel(title_, fontsize=_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=_fontsize)
    plt.legend(fontsize=15, loc=_loc)
    plt.tight_layout()
    if save:
        if file2save is None:
            print("Saving figure at ./figures/BPCpaper/%s_%s.png" % (param_str, val_str))
            plt.savefig('./figures/BPCpaper/%s_%s.png' % (param_str, val_str))
            with open('./figures/BPCpaper/%s_%s.txt' % (param_str, val_str), 'w') as file:
                file.write(file_string)
        else:
            print("Saving figure at %s%s_%s.png" % (file2save, param_str, val_str))
            plt.savefig('%s%s_%s.png' % (file2save, param_str, val_str))
            with open('%s%s_%s.txt' % (file2save, param_str, val_str), 'w') as file:
                file.write(file_string)
    else:
        if little_oh: # eps = o(tau^2)
            plt.title(title_ + r', $\epsilon = \tau^3, \gamma = \tau^\alpha$', fontsize=15)
        elif not little_oh and zero: # eps = 0
            plt.title(title_ + r', $\epsilon = 0, \gamma = \tau^\alpha$', fontsize=15)
        else: # eps = O(tau^2)
            plt.title(title_ + r', $\epsilon = \tau^2, \gamma = \tau^\alpha$', fontsize=15)
    plt.show()

    return







""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

################# Graph generation and other calculations ######################

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
def make_sim_graph(X, k_nn=5, sigma=3.):
    N = X.shape[0]
    # Make weighted similarity graph, in W
    D = sqdist(X.T,X.T)
    ind_knn = np.argsort(D, axis=1)[:,1:k_nn+1]
    Dknn = D[(np.arange(N).reshape(N,1),ind_knn)]

    I = np.tile(np.arange(N).reshape(N,1), (1,k_nn)).flatten()
    J = ind_knn.flatten()
    Dmean = (np.sum(Dknn, axis=1)/k_nn).reshape(N,1)
    w_sp = np.divide(Dknn, Dmean)
    w_sp = np.exp(-(w_sp * w_sp)/sigma)
    W = csr_matrix((w_sp.flatten() , (I, J)), shape=(N,N))
    W = 0.5*(W+W.T)
    return W



def make_sim_graph_pre(X, k_nn=5):
    N = X.shape[0]
    # Make weighted similarity graph, in W
    D = sqdist(X.T,X.T)
    ind_knn = np.argsort(D, axis=1)[:,1:k_nn+1]
    Dknn = D[(np.arange(N).reshape(N,1),ind_knn)]
    Dmean = (np.sum(Dknn, axis=1)/k_nn).reshape(N,1)
    Dmean[Dmean == 0.] = 1.
    w_sp = np.divide(Dknn, Dmean)
    return w_sp, ind_knn

def get_eig_Lnorm(W, num_eig=None, normed_=True):
    # If num_eig is None, we will calculate all eigenvalues
    if num_eig is None:
        num_eig = W.shape[0]
    L = csgraph.laplacian(W, normed=normed_).toarray()
    w, v = sla.eigh(L, eigvals=(0,num_eig-1))
    return w, v

def get_voting_data(sup_percent=0.1, filepath='datasets/VOTING-RECORD/', seed=None):
    filename = 'house-votes-84.data'
    np.random.seed(seed)
    # load the file directly into X format
    with open(filepath + filename, 'r') as f:
        lines = list(f)
        f.close()
    vote2num = {'y' : 1, 'n' : -1, '?' : 0} # maps y, n, ? to numerical values
    party2num = {'democrat' : 1, 'republican' : -1}
    X = [list(map(lambda x: vote2num[x], line.strip().split(',')[1:])) for line in lines]
    X = np.array(X)
    labels = [party2num[line.split(',')[0]] for line in lines]
    labels = np.array(labels)
    del lines
    # setup fid
    # -1 case
    fid = {}
    m1_ind = np.where(labels == -1)[0]
    np.random.shuffle(m1_ind)
    fid[-1] = list(m1_ind[:int(sup_percent*len(m1_ind))])
    # +1 case
    p1_ind = np.where(labels == 1)[0]
    np.random.shuffle(p1_ind)
    fid[1] = list(p1_ind[:int(sup_percent*len(p1_ind))])

    return X, labels, fid


def voting_run_test(SIGMAS, T, ALPHAS, rand=False, sup_percent=0.1, seed=None, k_nn=5, normalized=True):
    # Voting Record graph initializations
    X, labels, fid = get_voting_data(sup_percent=sup_percent, seed=seed)
    org_indices = np.argsort(labels)



    N = X.shape[0]
    w_sp0, ind_knn = make_sim_graph_pre(X.astype(np.float32), k_nn)
    I = np.tile(np.arange(N).reshape(N,1), (1,k_nn)).flatten()
    J = ind_knn.flatten()

    class1_ind = np.where(labels == 1)[0]
    class_1_ind = np.where(labels == -1)[0]
    labeled = np.concatenate((class1_ind, class_1_ind))
    sigma0 = SIGMAS[0]/2.  # smaller sigma -- give us a separated case (epsilon estimate closer to 0)?
    w_sp_0 = np.exp(-(w_sp0 * w_sp0)/sigma0) # I think this old code is actually squaring the already squared distances...?

    W_0 = csr_matrix((w_sp0.flatten(), (I, J)), shape=(N,N))
    W_0 = 0.5*(W_0+W_0.T)
    #plt.spy(W_0.toarray()[np.ix_(org_indices,org_indices)])
    #plt.show()
    W0 = lil_matrix((N,N), dtype=np.float32)
    W0[np.ix_(class1_ind, class1_ind)] = W_0[np.ix_(class1_ind, class1_ind)]
    W0[np.ix_(class_1_ind, class_1_ind)]= W_0[np.ix_(class_1_ind, class_1_ind)]
    #print()
    #plt.spy(W0.toarray()[np.ix_(org_indices, org_indices)])
    #plt.show()
    D_0 = np.array(np.sqrt(np.sum(W0, axis=1))).flatten()  #sqrt of the degrees

    # These Chis are scaled by the diagonal matrix of our "epsilon=0" case of a smaller sigma than any in SIGMAS
    Chis = np.zeros((N,2))
    Chis[class1_ind, 0] = 1./np.sqrt(len(class1_ind))
    Chis[class_1_ind, 1] = 1./np.sqrt(len(class_1_ind))
    if normalized:
        Chis[class1_ind, 0] *= D_0[class1_ind]
        Chis[class1_ind, 0] /= sp.linalg.norm(Chis[class1_ind, 0])
        Chis[class_1_ind, 1] *= D_0[class_1_ind]
        Chis[class_1_ind, 1] /= sp.linalg.norm(Chis[class_1_ind, 1])

    if rand: # each column is a random linear combination of columns of Chis
        u = Chis.dot(np.random.randn(2, 2))

    else:     # each column is a random scaling of corresponding columns of Chis
        u = Chis.dot(np.diag(np.random.randn(2)))


    u_n = u[:,0]/sp.linalg.norm(u[:,0])


    lab_ind = list(fid[1])
    lab_ind.extend(list(fid[-1]))
    diag = np.zeros(N)
    diag[lab_ind] = 1.
    B = sp.sparse.diags(diag, format='csr')

    # Data matrices for recording values
    TRC = np.zeros((len(ALPHAS), len(T), len(SIGMAS)))
    TRCBC = np.zeros((len(ALPHAS), len(T), len(SIGMAS)))
    BIAS = np.zeros((len(ALPHAS), len(T), len(SIGMAS)))

    EVALS = []

    for k,sigma in enumerate(SIGMAS):
        # Get weight matrix with given kernel width, sigma.
        #w_sp = np.exp(-w_sp0/sigma)
        w_sp = np.exp(-(w_sp0 * w_sp0)/sigma) # I think this old code is actually squaring the already squared distances...?
        W_s = csr_matrix((w_sp.flatten(), (I, J)), shape=(N,N))
        W_s = 0.5*(W_s+W_s.T)
        evals, evecs = get_eig_Lnorm(W_s, num_eig=None, normed_=normalized)
        EVALS.append(evals[1])
        # For each value of tau given in the list T
        for j, tau in enumerate(T):
            for i, alpha in enumerate(ALPHAS):
                gamma = tau**alpha
                d_inv = (tau ** (-2. * alpha)) * np.power(evals + tau**2., alpha)     # diagonalization of C_t,e - will be used later

                # prior_inv : C_{tau,eps}^{-1}, where
                # C_{tau, eps}^{-1} = tau^{-2alpha}(L + tau^2 I)^alpha
                prior_inv = evecs.dot(sp.sparse.diags(d_inv, format='csr').dot(evecs.T))
                # B/gamma^2
                B_over_gamma2 = B / (gamma**2.)
                # post_inv  : (B/gamma^2 + C_{tau,\eps}^{-1})^{-1}
                post_inv  = prior_inv + B_over_gamma2
                # C^{-1}
                post = sp.linalg.inv(post_inv)
                bias = sp.linalg.norm(post.dot(prior_inv.dot(u_n)))
                BIAS[i,j,k] = bias**2.

                # Calculate Tr(C)
                trC = sp.trace(post)
                TRC[i,j,k] = trC

                # Calculate Tr(CBC)/gamma^2
                post2 = post.dot(post)
                trCBC = sp.trace(post2[np.ix_(labeled, labeled)])
                TRCBC[i,j,k] = trCBC/(gamma**2.)
    print("Spectral gap evals are: %s" % str(EVALS))
    return TRC, TRCBC, BIAS

def voting_plot_data(T, data, ALPHAS, param_str, title_=r'$\mathrm{Tr}(C^*)$', val_str="TRC", save=False, Jval=-1, _fontsize=25):
    n_alpha = len(ALPHAS)
    if Jval >= 0:
        # Fit the line for T[J[i]:], TRC[i,:]
        print('Line fitting for %s' % val_str)
        J = n_alpha*[Jval]
        line_stats = np.zeros((n_alpha,2))
        for i in range(n_alpha):
            j = J[i]
            t = np.log(T[j:])
            A = np.array([t, np.ones(len(t))]).T
            b = np.log(data[i,j:])
            res = lsq_linear(A, b)
            line_stats[i,:] = res.x
            print('The slope for alpha = %1.1f is : %2.4f' % (ALPHAS[i], res.x[0]))

    markers = ['o','*', 'v', '^', '<', '>', '8', 's', 'p', 'h']
    fig = plt.figure()
    ax = fig.gca()
    ax.loglog(T, data.T)
    for i, alpha in enumerate(ALPHAS):
        ax.scatter(T, data[i,:], marker=markers[i], label=r'$\alpha =$ %2.1f'% alpha)
    plt.xlabel(r'$\tau$', fontsize=_fontsize)
    plt.ylabel(title_, fontsize=_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=_fontsize)
    if val_str == "BIAS":
        plt.legend(fontsize=15, loc='lower right')
    else:
        plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        print("Saving figure at ./figures/paper/%s_%s.png" % (param_str, val_str))
        plt.savefig('./figures/BPCpaper/%s_%s.png' % (param_str, val_str))
    else:
        plt.title(title_ + r', $\gamma = \tau^\alpha$', fontsize=15)
    plt.show()

    return

def voting_plot_data2(T, data, SIGMAS, param_str, title_=r'$\mathrm{Tr}(C^*)$', val_str="TRC", save=False, _fontsize=25):
    n_sigma = len(SIGMAS)
    markers = ['o','*', 'v', '^', '<', '>', '8', 's', 'p', 'h']
    fig = plt.figure()
    ax = fig.gca()
    ax.loglog(T, data)
    for i, sigma in enumerate(SIGMAS):
        ax.scatter(T, data[:,i], marker=markers[i], label=r'$\sigma =$ %2.1f'% sigma)
    plt.xlabel(r'$\tau$', fontsize=_fontsize)
    plt.ylabel(title_, fontsize=_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=_fontsize)
    if val_str == "BIAS":
        plt.legend(fontsize=15, loc='lower right')
    else:
        plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        print("Saving figure at ./figures/paper/%s_%s.png" % (param_str, val_str))
        plt.savefig('./figures/BPCpaper/%s_%s.png' % (param_str, val_str))
    else:
        plt.title(title_, fontsize=15)
    plt.show()

    return
