import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg
from scipy.optimize import lsq_linear
import scipy.linalg as sla
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import newton, root_scalar
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csgraph
import time
from heapq import *
from sklearn.datasets import make_moons
import copy
from itertools import permutations
from sklearn.datasets import make_moons



########################## Single Updates #######################

def calc_next_m(m, C, y, lab, k, y_k, gamma2):
    ck = C[k,:]
    ckk = ck[k]
    ip = np.dot(ck[lab], y[lab])
    val = ((gamma2)*y_k -ip )/(gamma2*(gamma2 + ckk))
    m_k = m + val*ck
    return m_k

def calc_next_C_and_m(m, C, y, lab, k, y_k, gamma2):
    ck = C[k,:]
    ckk = ck[k]
    ip = np.dot(ck[lab], y[lab])
    val = ((gamma2)*y_k -ip )/(gamma2*(gamma2 + ckk))
    m_k = m + val*ck

    # calculate C_k -- the posterior of adding k, y_k
    C_k = C - (1./(gamma2 + ckk))*np.outer(ck,ck)
    return m_k, C_k

def calc_next_C_and_m_multi(m, C, y, lab, k, class_ind_k, gamma2):
    ck = C[k,:]
    ckk = ck[k]
    ec = np.zeros(y.shape[1])
    ec[class_ind_k] = 1.
    ip = np.dot(ck[lab], y[lab])
    outer_term = (ec - (ip.T/gamma2))/(gamma2 + ckk)
    m_k = m + np.outer(ck,outer_term)

    # calculate C_k -- the posterior of adding k, y_k
    C_k = C - (1./(gamma2 + ckk))*np.outer(ck,ck)
    return m_k, C_k

def calc_next_m_batch(m, C, y, lab, k_to_add, y_ks, gamma2):
    C_b = C[:, k_to_add]
    lab_new = lab[:]
    lab_new.extend(k_to_add)
    y_next = y.copy()
    y_next[k_to_add] = y_ks
    m_next = m + C_b.dot(y_ks)/gamma2
    C_bb_inv = sla.inv(gamma2*np.eye(len(k_to_add)) + C_b[k_to_add,:])
    m_next -= (1./gamma2)*C_b.dot(C_bb_inv.dot(C_b[lab_new,:].T.dot(y_next[lab_new])))
    return m_next

'''
def calc_next_C_and_m_batch(m, C, y, lab, k_to_add, y_ks, gamma2):
    Cb = C[:,k_to_add]
    mat_inv = sla.inv(gamma2*np.eye(len(k_to_add)) + Cb[k_to_add,:])
    C -= Cb.dot(mat_inv.dot(Cb.T))

    # Update m now
    lab_new = lab[:]
    lab_new.extend(k_to_add)
    y_next = y.copy()
    y_next[k_to_add] = y_ks
    m_batch = (1./gamma2)*C[:,lab_new].dot(y_next[lab_new])

    return m_batch, C
'''


def calc_next_m_batch_multi(m, C, y, lab, k_to_add, class_ind_ks, gamma2):
    C_b = C[:, k_to_add]
    lab_new = lab[:]
    lab_new.extend(k_to_add)
    y_next = y.copy()
    y_next[np.ix_(k_to_add, class_ind_ks)] = 1.
    m_next = m + C_b.dot(y_next[k_to_add,:])/gamma2
    C_bb_inv = sla.inv(gamma2*np.eye(len(k_to_add)) + C_b[k_to_add,:])
    m_next -= (1./gamma2)*C_b.dot(C_bb_inv.dot(C_b[lab_new,:].T.dot(y_next[lab_new,:])))
    return m_next, y_next, lab_new

def calc_next_C_and_m_batch_multi(m, C, y, lab, k_to_add, class_ind_ks, gamma2):
    Cb = C[:,k_to_add]
    mat_inv = sla.inv(gamma2*np.eye(len(k_to_add)) + Cb[k_to_add,:])
    C -= Cb.dot(mat_inv.dot(Cb.T))

    # Update m now
    lab_new = lab[:]
    lab_new.extend(k_to_add)
    y_next = y.copy()
    y_next[np.ix_(k_to_add,class_ind_ks)] = 1.
    m_batch = (1./gamma2)*C[:,lab_new].dot(y_next[lab_new,:])

    return m_batch, C, y_next, lab_new

# Transform the vector m into probabilities, while still respecting the threshold value currently at 0
def get_probs(m,sigmoid=True):
    if sigmoid:
        return 1./(1. + np.exp(-3.*m))
    m_probs = m.copy()
    # simple fix to get probabilities that respect the 0 threshold
    m_probs[np.where(m_probs >0)] /= 2.*np.max(m_probs)
    m_probs[np.where(m_probs <0)] /= -2.*np.min(m_probs)
    m_probs += 0.5
    return m_probs


# Transform the matrix m into probabilities, while still respecting the threshold
def get_probs_multi(m, softmax=False):
    if m.size == m.shape[0]:
        return get_probs(m, softmax)

    if softmax:
        print('Trying softmax function...')
        m_probs = np.exp(3.*m)
        m_probs /= np.sum(m_probs, axis=1)
        return m_probs

    m_probs = m.copy()
    m_probs -= 0.5
    for j in range(m.shape[1]): # for each class vector, normalize to be probability in 0,1, respecting 0.5 threshold
        m_probs[np.where(m_probs[:,j] >0),j] /= 2.*np.max(m_probs[:,j])
        m_probs[np.where(m_probs[:,j] <0),j] /= -2.*np.min(m_probs[:,j])
    m_probs += 0.5
    return m_probs


############### EEM Risk #########
def calc_risk(k, m, C, y, lab, unlab, m_probs, gamma2):
    m_at_k = m_probs[k]
    m_k_p1 = calc_next_m(m, C, y, lab, k, 1., gamma2)
    m_k_p1 = get_probs(m_k_p1)
    risk = m_at_k*np.sum([min(m_k_p1[i], 1.- m_k_p1[i]) for i in unlab])
    m_k_m1 = calc_next_m(m, C, y, lab, k, -1., gamma2)
    m_k_m1 = get_probs(m_k_m1)
    risk += (1.-m_at_k)*np.sum([min(m_k_m1[i], 1.- m_k_m1[i]) for i in unlab])
    return risk

def calc_risk_full(k, m, C, y, lab, unlab, m_probs, gamma2):
    N = C.shape[0]
    m_at_k = m_probs[k]
    m_k_p1 = calc_next_m(m, C, y, lab, k, 1., gamma2)
    m_k_p1 = get_probs(m_k_p1)
    risk = m_at_k*np.sum([min(m_k_p1[i], 1.- m_k_p1[i]) for i in range(N)])
    m_k_m1 = calc_next_m(m, C, y, lab, k, -1., gamma2)
    m_k_m1 = get_probs(m_k_m1)
    risk += (1.-m_at_k)*np.sum([min(m_k_m1[i], 1.- m_k_m1[i]) for i in range(N)])
    return risk

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


def run_next_EEM(m, C, y, lab, unlab, fid, ground_truth, gamma2, verbose=False, risk_full=False):
    tic = time.clock()
    risks = []
    m_probs = get_probs_multi(m)
    for j in unlab:
        if risk_full:
            risk_j = calc_risk_full(j, m, C, y, lab, unlab, m_probs, gamma2)
        else:
            risk_j = calc_risk(j, m, C, y, lab, unlab, m_probs, gamma2)
        heappush(risks, (risk_j, j))

    k_next_risk, k_next = heappop(risks)
    toc = time.clock()
    if verbose:
        print('Time for EEM = %f' % (toc - tic))

    # Ask "the oracle" for k_next's value, known from ground truth in Ns
    y_k_next = ground_truth[k_next]
    fid[y_k_next].append(k_next)

    m_next, C_next = calc_next_C_and_m(m, C, y, lab, k_next, y_k_next, gamma2)
    y[k_next] = y_k_next
    lab.append(k_next)
    unlab.remove(k_next)


    return k_next, m_next, C_next, y, lab, unlab, fid

def V_opt(C, unlabeled, gamma2):
    ips = np.array([np.inner(C[k,:], C[k,:]) for k in unlabeled]).flatten()
    v_opt = ips/(gamma2 + np.diag(C)[unlabeled])
    k_max = unlabeled[np.argmax(v_opt)]
    return k_max

def Sigma_opt(C, unlabeled, gamma2):
    sums = np.sum(C[np.ix_(unlabeled,unlabeled)], axis=1)
    sums = np.asarray(sums).flatten()**2.
    s_opt = sums/(gamma2 + np.diag(C)[unlabeled])
    k_max = unlabeled[np.argmax(s_opt)]
    return k_max


def run_next_VS_multi(m, C, y, labeled, unlabeled, fid, ground_truth, gamma2, method='S', batch_size=5, verbose=False):
    k_to_add = []
    C_next = C.copy()
    for i in range(batch_size):
        tic = time.clock()
        if method == 'V':
            k_next = V_opt(C_next, unlabeled, gamma2)
        elif method == 'S':
            k_next = Sigma_opt(C_next, unlabeled, gamma2)
        else:
            raiseValueError('Parameter for "method" is not valid...')
        toc = time.clock()
        if verbose:
            print('Time for %s_opt = %f' % (method, (toc - tic)))

        k_to_add.append(k_next)
        unlabeled.remove(k_next)  # we are updating unlabeled here

        # calculate update of C -- the posterior of adding k
        ck = C_next[k_next,:]
        ckk = ck[k_next]
        C_next -= (1./(gamma2 + ckk))*np.outer(ck,ck)


    if -1 in fid.keys():
        # Ask "the oracle" for values of the k in k_to_add value known from ground truth
        y_ks = [ground_truth[k] for k in k_to_add]

        # Do BATCH calculation now that we've queried the oracle, notice this is using the OLD C
        # just had found it was a little bit faster. could just do (1/gamma2)*C_next.dot(y[labeled_new])
        m_next = calc_next_m_batch(m, C, y, labeled, k_to_add, y_ks, gamma2)
        # update the observations vector y, labeled, and fid
        y_next = y.copy()
        y_next[k_to_add] = y_ks
        labeled.extend(k_to_add)

    else:
        # Ask "the oracle" for values of the k in k_to_add value known from ground truth
        ofs = min(list(fid.keys()))
        class_ind_ks = [ground_truth[k]-ofs for k in k_to_add]

        # Do BATCH calculation now that we've queried the oracle, notice this is using the OLD C
        # just had found it was a little bit faster. could just do (1/gamma2)*C_next.dot(y[labeled_new])
        m_next, y_next, labeled = calc_next_m_batch_multi(m, C, y, labeled, k_to_add, class_ind_ks, gamma2)

    del m, C, y  # delete now that no longer need

    # update fid
    for k in k_to_add:
        fid[ground_truth[k]].append(k)


    return m_next, C_next, y_next, labeled, unlabeled, fid, k_to_add



def run_test_AL_VS_multi(X, v, w, fid, ground_truth, method='S', tag2=(0.01, 1.0, 0.01), test_opts=(5, 10, False)):
    '''
    Inputs:
        X : (N x d) data matrix with the data points as columns
        v : (N x N) eigenvectors (as columns)
        w : (N, ) eigenvalues numpy array
        fid : dictionary with fidelity indices (class_i, [i_1, i_2, ...])
        ground_truth :
        method : Either 'S' (Sigma_opt) or 'V'(V_opt)
        tag2 : tuple (tau, alpha, gamma2). Default tau = 0.01, alpha = 1.0, gamma2 = 0.00001
        test_opts : tuple (batch_size, iters, verbose). Default (5, 10, False)
                (batch_size int, iters int, verbose bool)
                Note if iters % batch_size != 0 then we make iters to be a multiple of batch_size
    '''
    N = len(ground_truth)
    tau, alpha, gamma2 = tag2
    batch_size, iters, verbose = test_opts

    mod = iters % batch_size
    if mod != 0:
        iters += (batch_size - mod)
    num_batches = int(iters / batch_size)


    # Prepare datastructures for labeling uses later
    gt_flipped = {}
    indices = np.array(list(range(N)))
    labeled = set()
    for k in fid.keys():
        k_mask = indices[ground_truth ==k]
        gt_flipped[k] = k_mask
        labeled = labeled.union(set(fid[k]))
    unlabeled = sorted(list(set(indices) - labeled))
    labeled = sorted(list(labeled))

    # Initial solution - find m and C, keep track of y
    tic = time.clock()
    m, C, y = calc_orig_multi(v, w, fid, labeled, unlabeled, tau, alpha, gamma2)
    toc = time.clock()
    if verbose:
        print('calc_orig_multi took %f seconds' % (toc -tic))

    # Calculate the error of the classification resulting from this initial solution
    ERRS = []
    error, stats_obj = calc_stats_multi(m, fid, gt_flipped)
    ERRS.append((-1,error))
    if verbose:
        print('Iter = 0')
        if -1 in fid.keys():
            plot_iter(stats_obj, X, k_next=-1)
        else:
            plot_iter_multi(stats_obj, X, fid, k_next=-1)
    # structure to record the m vectors calculated at each iteration
    M = {}
    M[-1] = m

    # AL choices - done in a batch
    for l in range(num_batches):
        m, C, y, labeled, unlabeled, fid, k_added = run_next_VS_multi(m, C, y, labeled,
                            unlabeled, fid, ground_truth, gamma2, method, batch_size, verbose)
        error, stats_obj = calc_stats_multi(m, fid, gt_flipped)
        ERRS.append((k_added,error))
        M[l] = m
        if verbose:
            print('Iter = %d' % (l + 1))
            if -1 in fid.keys():
                plot_iter(stats_obj, X, k_next=k_added)
            else:
                plot_iter_multi(stats_obj, X, fid, k_next=k_added)
    return ERRS, M

def run_test_rand_multi(X, v, w, fid, ground_truth, tag2=(0.01, 1.0, 0.01), test_opts=(10, False), show_all_iters=False):
    '''
    Inputs:
        X : (N x d) data matrix with the data points as columns
        v : (N x N) eigenvectors (as columns)
        w : (N, ) eigenvalues numpy array
        fid : dictionary with fidelity indices (class_i, [i_1, i_2, ...])
        ground_truth :
        method : Either 'S' (Sigma_opt) or 'V'(V_opt)
        tag2 : tuple (tau, alpha, gamma2). Default tau = 0.01, alpha = 1.0, gamma2 = 0.00001
        test_opts : tuple (iters, verbose). Default (10, False)
                (iters int, verbose bool)
        show_all_iters : bool whether or not to calculate the error/plot at each single choice from random sampling
    '''
    N = len(ground_truth)
    tau, alpha, gamma2 = tag2
    iters, verbose = test_opts


    # Prepare datastructures for labeling uses later
    gt_flipped = {}
    indices = np.array(list(range(N)))
    labeled = set()
    for k in fid.keys():
        k_mask = indices[ground_truth ==k]
        gt_flipped[k] = k_mask
        labeled = labeled.union(set(fid[k]))
    unlabeled = sorted(list(set(indices) - labeled))
    labeled = sorted(list(labeled))

    classes = list(fid.keys())
    ofs = min(classes)

    # Initial solution - find m and C, keep track of y
    tic = time.clock()
    m, C, y = calc_orig_multi(v, w, fid, labeled, unlabeled, tau, alpha, gamma2)
    toc = time.clock()
    if verbose:
        print('calc_orig_multi took %f seconds' % (toc -tic))

    # Calculate the error of the classification resulting from this initial solution
    ERRS = []
    error, stats_obj = calc_stats_multi(m, fid, gt_flipped)
    ERRS.append((-1,error))
    if verbose:
        print('Iter = 0')
        if -1 in fid.keys():
            plot_iter(stats_obj, X, k_next=-1)
        else:
            plot_iter_multi(stats_obj, X, fid, k_next=-1)

    # structure to record the m vectors calculated at each iteration
    M = {}
    M[-1] = m

    # rand choices - show each or all at once
    if show_all_iters:
        for l in range(iters):
            k_next = np.random.choice(unlabeled,1)[0]
            if -1 in fid.keys():
                y_k_next = ground_truth[k_next]
                y[k_next] = y_k_next
                m, C = calc_next_C_and_m(m, C, y, labeled, k_next, y_k_next, gamma2)
                labeled = np.array(list(labeled)+ [k_next])
            else:
                class_ind_k_next = ground_truth[k_next]-ofs
                y[k_next,class_ind_k_next] = 1.
                m, C = calc_next_C_and_m_multi(m, C, y, labeled, k_next, class_ind_k_next, gamma2)
                labeled.append(k_next)

            unlabeled.remove(k_next)

            M[l] = m
            error, stats_obj = calc_stats_multi(m, fid, gt_flipped)
            ERRS.append((k_next, error))
            if verbose:
                print('Iter = %d' % (l + 1))
                if -1 in fid.keys():
                    plot_iter(stats_obj, X, k_next=k_next)
                else:
                    plot_iter_multi(stats_obj, X, fid, k_next=k_next)
    else:
        k_to_add = list(np.random.choice(unlabeled,iters, replace=False))
        for k in k_to_add:
            unlabeled.remove(k)

        if -1 in fid.keys():
            # Ask "the oracle" for values of the k in k_to_add value known from ground truth
            y_ks = [ground_truth[k] for k in k_to_add]
            # Do BATCH calculation now that we've queried the oracle, notice this is using the OLD C
            # just had found it was a little bit faster. could just do (1/gamma2)*C_next.dot(y[labeled_new])
            m_next = calc_next_m_batch(m, C, y, labeled, k_to_add, y_ks, gamma2)
            # update the observations vector y, labeled, and fid
            y[k_to_add] = y_ks
            labeled.extend(k_to_add)

        else:
            class_ind_ks = [ground_truth[k]-ofs for k in k_to_add]
            m, C, y, labeled = calc_next_C_and_m_batch_multi(m, C, y, labeled, k_to_add, class_ind_ks, gamma2)

        error, stats_obj = calc_stats_multi(m, fid, gt_flipped)
        ERRS.append((k_to_add,error))
        M[0] = m
        if verbose:
            if -1 in fid.keys():
                plot_iter(stats_obj, X, k_next=k_to_add)
            else:
                plot_iter_multi(stats_obj, X, fid, k_next=k_to_add)

    return ERRS, M

# BINARY CASE ONLY - since for EEM
def run_test_AL(X, v, w, fid, ground_truth, tag2=(0.01, 1.0, 0.01), test_opts=(10, False)):
    '''
    Inputs:
        X : (N x d) data matrix with the data points as columns
        v : (N x N) eigenvectors (as columns)
        w : (N, ) eigenvalues numpy array
        fid : dictionary with fidelity indices (class_i, [i_1, i_2, ...])
        tag2 : tuple (tau, alpha, gamma2). Default tau = 0.01, alpha = 1.0, gamma2 = 0.00001
        test_opts : tuple (iters, verbose). Default (10, False)
                (iters int, verbose bool)
    '''
    N = len(ground_truth)
    tau, alpha, gamma2 = tag2
    iters, verbose = test_opts

    # Prepare datastructures for labeling uses later
    gt_flipped = {}
    indices = np.array(list(range(N)))
    labeled = set()
    for k in fid.keys():
        k_mask = indices[ground_truth ==k]
        gt_flipped[k] = k_mask
        labeled = labeled.union(set(fid[k]))
    unlabeled = sorted(list(set(indices) - labeled))
    labeled = sorted(list(labeled))

    org_indices = list(np.where(ground_truth == -1)[0])
    org_indices.extend(list(np.where(ground_truth == 1)[0]))

    # Initial solution - find m and C, keep track of y
    tic = time.clock()
    m, C, y = calc_orig_multi(v, w, fid, labeled, unlabeled, tau, alpha, gamma2)
    toc = time.clock()
    if verbose:
        print('calc_orig_multi took %f seconds' % (toc -tic))



    # Calculate the error of the classification resulting from this initial solution
    ERRS = []
    error, stats_obj = calc_stats_multi(m, fid, gt_flipped)
    ERRS.append((-1,error))
    if verbose:
        print('Iter = 0')
        if -1 in fid.keys():
            plot_iter(stats_obj, X, k_next=-1)
        else:
            plot_iter_multi(stats_obj, X, fid, k_next=-1)
        plot_risk_smoothness_and_m(X, C, m, y, labeled, unlabeled, org_indices, gamma2)

    # structure to record the m vectors calculated at each iteration
    M = {}
    M[-1] = m

    # AL choices
    for l in range(iters):
        k, m, C, y, labeled, unlabeled, fid = run_next_EEM(m, C, y, labeled, unlabeled, fid, ground_truth, gamma2)

        error, stats_obj = calc_stats_multi(m, fid, gt_flipped)
        ERRS.append((k,error))
        M[l] = m
        if verbose:
            print('Iter = %d' % (l + 1))
            if -1 in fid.keys():
                plot_iter(stats_obj, X, k_next=k)
            else:
                plot_iter_multi(stats_obj, X, fid, k_next=k)
            plot_risk_smoothness_and_m(X, C, m, y, labeled, unlabeled, org_indices, gamma2)
    return ERRS, M



def plot_risk_smoothness(X, C, m, y, labeled, unlabeled, gamma2):
    N = X.shape[0]
    m_probs = get_probs_multi(m)
    risks = [calc_risk(j, m, C, y, labeled, unlabeled, m_probs, gamma2) for j in range(N)]
    #risks2 = [calc_risk_full(j, m, C, y, labeled, unlabeled, m_probs, gamma2) for j in range(N)]
    imax,imin = np.argmax(risks), np.argmax(risks)
    val = (risks - min(risks))/(max(risks) - min(risks))
    colors = [(x, 0.5,(1-x)) for x in val]
    plt.scatter(X[:,0],X[:,1], c=colors)
    plt.scatter(X[imin,0], X[imin,1],c=[(0,0.5,1)], marker='v', label='low risk')
    plt.scatter(X[imin,0], X[imin,1], c= [(1,0.5,0)], marker='^', label='high risk')
    plt.legend()
    plt.title('Heat map of EEM function on nodes of graph')
    plt.show()
    print('k_next = %d' % np.argmin(risks))
    print('m[k_next] = %f' % m[np.argmin(risks)])
    return

def plot_risk_smoothness_and_m(X, C, m, y, labeled, unlabeled, org_indices, gamma2):
    N = X.shape[0]
    m_probs = get_probs_multi(m)
    risks = [calc_risk(j, m, C, y, labeled, unlabeled, m_probs, gamma2) for j in range(N)]
    imax,imin = np.argmax(risks), np.argmax(risks)
    #risks2 = [calc_risk_full(j, m, C, y, labeled, unlabeled, m_probs, gamma2) for j in range(N)]
    val = (risks - min(risks))/(max(risks) - min(risks))
    colors = [(x, 0.5,(1-x)) for x in val]
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.scatter(X[:,0],X[:,1], c=colors)
    plt.scatter(X[imin,0], X[imin,1],c=[(0,0.5,1)], label='low risk')
    plt.scatter(X[imin,0], X[imin,1], c= [(1,0.5,0)],label='high risk')
    plt.legend()
    plt.title('Heat map of EEM function on nodes of graph')
    plt.subplot(1,2,2)
    lin = [i for i in range(N)]
    plt.scatter(lin, get_probs(m[org_indices]))
    plt.plot(lin, N*[0.5], 'r--')
    plt.title(r'$m$ probabilities')
    plt.show()
    return

def plot_m(X, m):
    N = X.shape[0]
    val = (m - min(m))/(max(m) - min(m))
    colors = [(x,(1-x),0.5) for x in val]
    plt.scatter(X[:,0],X[:,1], c=colors)
    plt.scatter(2,1.45,c=[(0,1,0.5)], label='most negative')
    plt.scatter(-1,1.45, c= [(1,0,0.5)], label='most positive')
    plt.legend()
    plt.title('Heat map of mean m on nodes of graph')
    plt.show()
    return
