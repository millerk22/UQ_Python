import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import lsq_linear
import scipy.linalg as sla
import numpy as np 
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import newton, root_scalar
from mpl_toolkits.mplot3d import Axes3D


def generate_random_graphs(Ns, eps, perturb_level, seed, keep_W_0=False):
    '''
    inputs   : 
      Ns    : class sizes, an array of length K
      eps   : epsilon
      perturb_level  : W = W_0 + eps W_1 + ... + eps^level W_level  
      seed  : random seed
    returns  : adjacency matrix 
    '''
    W0s = []
    density = 1.0
    for Ni in Ns:
        W0s.append(sp.sparse.random(Ni, Ni, density=density, random_state=seed))
    W0 = sp.sparse.block_diag(W0s, format='csr')
    if keep_W_0:
        W_orig = W0.copy()
    shape = W0.shape
    for i in range(perturb_level):
        W0 += eps * (eps**i) \
         * sp.sparse.random(*shape, density=density, format='csr', 
                           random_state=seed)
    # make sure W is symmetric
    W0 += W0.T
    W0 /= 2
    # remove diagonals
    W0 -= sp.sparse.diags(W0.diagonal())
    if keep_W_0:
        return W0, W_orig
    else:
        return W0


def compute_laplacian(W):
    D = sp.sparse.diags(np.sum(W, axis=0).getA().flatten())
    L = D - W
    return L, D.diagonal()

# Main setup, defining the node set, fidelity nodes, ground truth u and observations y, and corresponding
# B and H matrices

def overall_setup(rand=False, Ns=[100,100,100], num_in_each=5):  
    N = sum(Ns)

    # begin with ground truth, u
    u = np.zeros((N,len(Ns)))
    fid = np.zeros(N, dtype='int')

    Chis = np.zeros((N,len(Ns)))
    idx = 0
    for l in range(len(Ns)):
        fid[idx:idx+num_in_each] = 1
        Chis[idx:idx+Ns[l],l] = 1./np.sqrt(Ns[l])   
        idx += Ns[l]
    
    
    if rand: # each column is a random linear combination of columns of Chis
        u = Chis.dot(np.random.randn(len(Ns), len(Ns)))
        
    else:     # each column is a random scaling of corresponding columns of Chis
        u = Chis.dot(np.diag(np.random.randn(len(Ns))))
        
    B = sp.sparse.diags(fid, format='csr')
    labeled = np.nonzero(fid)

    H = np.zeros((len(labeled[0]), N))
    for i in range(len(labeled[0])):
        H[i,labeled[0][i]] = 1.
    
    draws = np.random.randn(len(Ns),len(labeled[0]))  # get draws for the observations y, dependent on gamma later on
    
    
    return Ns, u, B, labeled, H, draws



from scipy.sparse import csgraph

def get_eig_Lnorm(eps, Ns):    
    # Calculate the matrices to be involved. ensure we draw the same graph each time for consistency
    W = generate_random_graphs(Ns, eps, perturb_level=3, seed=10)
    L_sym = csgraph.laplacian(W, normed=True)
    [w, v] = sp.linalg.eigh(L_sym.toarray())
    
    return w, v


def get_eig(eps, Ns):    
    # Calculate the matrices to be involved. ensure we draw the same graph each time for consistency
    W = generate_random_graphs(Ns, eps, perturb_level=3, seed=10)
    L,D = compute_laplacian(W)
    [w, v] = sp.linalg.eigh(L.toarray())
    
    return w, v


def run_test(T, ALPHAS, Ns, B, labeled, little_oh=False):
    TRC = np.zeros((len(ALPHAS), len(T)))
    TRCBC = np.zeros((len(ALPHAS), len(T)))
    for j, tau in enumerate(T):
        if little_oh:
            eps = tau**3.
        else:
            eps = tau**2.
            
        w,v = get_eig_Lnorm(eps, Ns)
        #w,v = get_eig(eps, Ns)
        for i,alpha in enumerate(ALPHAS):
            gamma = tau**alpha
            d = (tau ** (2 * alpha)) * np.power(w + tau**2., -alpha)     # diagonalization of C_t,e

            # prior_inv : C_{tau,eps}^{-1}, where
            # C_{tau, eps}^{-1} = tau^{-2alpha}(L + tau^2 I)^alpha 
            prior_inv = v.dot(sp.sparse.diags([1./thing for thing in d], format='csr').dot(v.T))
            # B/gamma^2
            B_over_gamma2 = B / (gamma * gamma)
            # post_inv  : (B/gamma^2 + C_{tau,\eps}^{-1})^{-1}
            post_inv  = prior_inv + B_over_gamma2
            # C^{-1}
            post = post_inv.I
            trC = sp.trace(post)
            TRC[i,j] = trC

            post2 = post.dot(post)
            trCBC = sp.trace(post2[np.ix_(labeled[0], labeled[0])])
            TRCBC[i,j] = trCBC/(gamma**2.)
    
    return TRC, TRCBC

def plot_trC(T, TRC, ALPHAS, param_str, save=False, Jval=-1,_fontsize=25):
    n_alpha = len(ALPHAS)
    if Jval >= 0:
        # Fit the line for T[J[i]:], TRC[i,:]
        print('Line fitting for Tr(C)')
        J = n_alpha*[Jval]
        line_stats = np.zeros((n_alpha,2))
        for i in range(n_alpha):
            j = J[i]
            t = np.log(T[j:])
            A = np.array([t, np.ones(len(t))]).T
            b = np.log(TRC[i,j:])
            res = lsq_linear(A, b)
            line_stats[i,:] = res.x
            print('The slope for alpha = %1.1f is : %2.4f' % (ALPHAS[i], res.x[0]))
    
    markers = ['o','*', 'v', '^', '<', '>', '8', 's', 'p', 'h']
    fig = plt.figure()
    ax = fig.gca()
    ax.loglog(T, TRC.T)
    for i, alpha in enumerate(ALPHAS):
        ax.scatter(T, TRC[i,:], marker=markers[i], label=r'$\alpha =$ %2.1f'% alpha)
    plt.xlabel(r'$\tau$', fontsize=_fontsize)
    plt.ylabel(r'$\mathrm{Tr}(C^*)$', fontsize=_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=_fontsize)
    #plt.legend([r'$\alpha =$ %2.1f'% alpha for alpha in ALPHAS], fontsize=20)
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig('./figures/paper/%s_C.png' % param_str)
    else:
        plt.title(r'$\mathrm{Tr}(C^*), \epsilon = \tau^2, \gamma = \tau^\alpha$ ', fontsize=15)
    plt.show()
    
    return

def plot_trCBC(T, TRCBC, ALPHAS, param_str, save=False, Jval=-1,_fontsize=25):
    n_alpha = len(ALPHAS)
    if Jval >= 0:
        # Fit the line for T[J[i]:], TRC[i,:]
        print('Line fitting for Tr(CBC)/gamma^2')
        J = n_alpha*[6]
        line_stats = np.zeros((n_alpha,2))
        for i in range(n_alpha):
            j = J[i]
            t = np.log(T[j:])
            A = np.array([t, np.ones(len(t))]).T
            b = np.log(TRCBC[i,j:])
            res = lsq_linear(A, b)
            line_stats[i,:] = res.x
            print('The slope for alpha = %1.1f is : %2.4f' % (ALPHAS[i], res.x[0]))
            
            
    markers = ['o','*', 'v', '^', '<', '>', '8', 's', 'p', 'h']
    fig = plt.figure()
    ax = fig.gca()
    ax.loglog(T, TRCBC.T)
    for i, alpha in enumerate(ALPHAS):
        ax.scatter(T, TRCBC[i,:], marker=markers[i], label=r'$\alpha =$ %2.1f'% alpha)
    plt.xlabel(r'$\tau$', fontsize=_fontsize)
    plt.ylabel(r'$\frac{1}{\gamma^2}\mathrm{Tr}(C^*BC^*)$', fontsize=_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=_fontsize)
    #plt.legend([r'$\alpha =$ %2.1f'% alpha for alpha in ALPHAS], fontsize=20)
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig('./figures/paper/%s_CBC.png' % param_str)
    else:
        plt.title(r'$\frac{1}{\gamma^2}\mathrm{Tr}(C^*BC^*), \epsilon = \tau^2, \gamma = \tau^\alpha$', fontsize=15)
    plt.show()
    
    return
    
    
    
def run_test2(T, ALPHAS, Ns, B, labeled, u, little_oh=False):
    TRC = np.zeros((len(ALPHAS), len(T)))
    TRCBC = np.zeros((len(ALPHAS), len(T)))
    normC = np.zeros((len(ALPHAS), len(T)))
    normC_tau_inv_u = np.zeros((len(ALPHAS), len(T)))
    normC_tau_inv_u_p1 = np.zeros((len(ALPHAS), len(T)))
    normC_tau_inv_u_p2 = np.zeros((len(ALPHAS), len(T)))
    BIAS = np.zeros((len(ALPHAS), len(T)))
    
    K = len(Ns)
    u_n = u[:,0]/sp.linalg.norm(u[:,0])
    for j, tau in enumerate(T):
        if little_oh:
            eps = tau**3.
        else:
            eps = tau**2.
            
        #w,v = get_eig_Lnorm(eps, Ns)
        w,v = get_eig(eps, Ns)
        for i,alpha in enumerate(ALPHAS):
            gamma = tau**alpha
            d = (tau ** (2 * alpha)) * np.power(w + tau**2., -alpha)     # diagonalization of C_t,e
            # prior_inv : C_{tau,eps}^{-1}, where
            # C_{tau, eps}^{-1} = tau^{-2alpha}(L + tau^2 I)^alpha 
            prior_inv = v.dot(sp.sparse.diags([1./thing for thing in d], format='csr').dot(v.T))
            
            
            normC_tau_inv_u[i,j] = sp.linalg.norm(prior_inv.dot(u_n))
            vK = v[:,:K]
            u_p1 = vK.dot(vK.T.dot(u_n))
            u_p2 = u_n - u_p1
            normC_tau_inv_u_p1[i,j] = sp.linalg.norm(prior_inv.dot(u_p1))
            normC_tau_inv_u_p2[i,j] = sp.linalg.norm(prior_inv.dot(u_p2))
            
            # B/gamma^2
            B_over_gamma2 = B / (gamma * gamma)
            # post_inv  : (B/gamma^2 + C_{tau,\eps}^{-1})^{-1}
            post_inv  = prior_inv + B_over_gamma2
            # C^{-1}
            post = post_inv.I
            bias = sp.linalg.norm(post.dot(prior_inv.dot(u_n)))
            BIAS[i,j] = bias
            normC[i,j] = sp.linalg.norm(post)
            
            trC = sp.trace(post)
            TRC[i,j] = trC

            post2 = post.dot(post)
            trCBC = sp.trace(post2[np.ix_(labeled[0], labeled[0])])
            TRCBC[i,j] = trCBC/(gamma**2.)
    
    return TRC, TRCBC, normC, normC_tau_inv_u, normC_tau_inv_u_p1, normC_tau_inv_u_p2, BIAS

def plot_data(T, data, ALPHAS, param_str, title_=r'$\mathrm{Tr}(C^*)$', save=False, Jval=-1,_fontsize=25):
    n_alpha = len(ALPHAS)
    if Jval >= 0:
        # Fit the line for T[J[i]:], TRC[i,:]
        print('Line fitting for Tr(C)')
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
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig('./figures/paper/%s_C.png' % param_str)
    else:
        plt.title(title_ + r', $\epsilon = \tau^2, \gamma = \tau^\alpha$ ', fontsize=15)
    plt.show()
    
    return
    
    




def generate_random_graphs_rest(Ns, eps, W_0, perturb_level, seed):
    '''
    inputs   : 
      Ns    : class sizes, an array of length K
      eps   : epsilon
      W_0   : the perfectly separated weight matrix, will be common among all iterations
      perturb_level  : W = W_0 + eps W_1 + ... + eps^level W_level  
      seed  : random seed
    returns  : adjacency matrix 
    '''
    density = 1.0
    W0 = W_0.copy()
    shape = W0.shape
    for i in range(perturb_level):
        W0 += eps * (eps**i) \
         * sp.sparse.random(*shape, density=density, format='csr', 
                           random_state=seed)
    # make sure W is symmetric
    W0 += W0.T
    W0 /= 2
    # remove diagonals
    W0 -= sp.sparse.diags(W0.diagonal())
    
    return W0


def get_eig_Lnorm_withW0(eps, Ns, W_0):    
    # Calculate the matrices to be involved. ensure we draw the same graph each time for consistency
    W = generate_random_graphs_rest(Ns, eps, W_0, perturb_level=3, seed=10)
    L_sym = csgraph.laplacian(W, normed=True)
    [w, v] = sp.linalg.eigh(L_sym.toarray())
    return w, v

def overall_setup_norm(rand=False,Ns=[100,100,100],num_in_each=5):  
    N = sum(Ns)
    W_0 = generate_random_graphs(Ns, 0., perturb_level=0, seed=10)
    D_0 = np.asarray(np.sum(W_0, axis=1)).flatten()**0.5   # normalization for the 
    print(D_0.shape)
    # begin with ground truth, u 
    u = np.zeros((N,len(Ns)))
    fid = np.zeros(N, dtype='int')

    Chis = np.zeros((N,len(Ns)))
    idx = 0
    for l in range(len(Ns)):
        fid[idx:idx+num_in_each] = 1
        Chis[idx:idx+Ns[l],l] = 1./np.sqrt(Ns[l])
        Chis[idx:idx+Ns[l],l] *= D_0[idx:idx+Ns[l]]
        Chis[idx:idx+Ns[l],l] /= sp.linalg.norm(Chis[idx:idx+Ns[l],l])
        idx += Ns[l]
    
    if rand: # each column is a random linear combination of columns of Chis
        u = Chis.dot(np.random.randn(len(Ns), len(Ns)))
        
    else:     # each column is a random scaling of corresponding columns of Chis
        u = Chis.dot(np.diag(np.random.randn(len(Ns))))
    
    B = sp.sparse.diags(fid, format='csr')
    labeled = np.nonzero(fid)

    H = np.zeros((len(labeled[0]), N))
    for i in range(len(labeled[0])):
        H[i,labeled[0][i]] = 1.
    
    draws = np.random.randn(len(Ns),len(labeled[0]))  # get draws for the observations y, dependent on gamma later on
    
    return Ns, u, B, labeled, H, draws, W_0



def run_test_norm(T, ALPHAS, Ns, B, labeled, u, W_0, little_oh=False):
    TRC = np.zeros((len(ALPHAS), len(T)))
    TRCBC = np.zeros((len(ALPHAS), len(T)))
    normC = np.zeros((len(ALPHAS), len(T)))
    normC_tau_inv_u = np.zeros((len(ALPHAS), len(T)))
    normC_tau_inv_u_p1 = np.zeros((len(ALPHAS), len(T)))
    normC_tau_inv_u_p2 = np.zeros((len(ALPHAS), len(T)))
    BIAS = np.zeros((len(ALPHAS), len(T)))
    
    K = len(Ns)
    N = sum(Ns)
    u_n = u[:,0]/sp.linalg.norm(u[:,0])
    print(np.linalg.norm(u_n))
    for j, tau in enumerate(T):
        if little_oh:
            eps = tau**3.
        else:
            eps = tau**2.
            
        
        w,v = get_eig_Lnorm_withW0(eps, Ns, W_0)
        
        for i,alpha in enumerate(ALPHAS):
            gamma = tau**alpha
            d = (tau ** (2 * alpha)) * np.power(w + tau**2., -alpha)     # diagonalization of C_t,e - will be used later
            
            
            d_inv = 1./d  # the eigenvalues of C_\tau,\eps^{-1}
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
            prior_inv = v.dot(sp.sparse.diags([1./thing for thing in d], format='csr').dot(v.T))
            # B/gamma^2
            B_over_gamma2 = B / (gamma * gamma)
            # post_inv  : (B/gamma^2 + C_{tau,\eps}^{-1})^{-1}
            post_inv  = prior_inv + B_over_gamma2
            # C^{-1}
            post = post_inv.I
            bias = sp.linalg.norm(post.dot(prior_inv.dot(u_n)))
            BIAS[i,j] = bias
            normC[i,j] = sp.linalg.norm(post)
            
            trC = sp.trace(post)
            TRC[i,j] = trC

            post2 = post.dot(post)
            trCBC = sp.trace(post2[np.ix_(labeled[0], labeled[0])])
            TRCBC[i,j] = trCBC/(gamma**2.)
    
    return TRC, TRCBC, normC, normC_tau_inv_u, normC_tau_inv_u_p1, normC_tau_inv_u_p2, BIAS