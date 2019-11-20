from datasets.data_loaders_mlflow import load_voting_records
from util.mlflow_util import get_prev_run, load_uri
import mlflow
import numpy as np
import scipy as sp
from datasets.Graph_manager import Graph_manager

def compute_groundtruth_u(X, labels, params):
    sigma = params['sigma']
    knn = params['knn']
    Ltype = params['Ltype']
    zp_k = params['zp_k']
    gm = Graph_manager()
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == -1)[0]
    W = gm.compute_similarity_graph(X, knn, sigma, zp_k) 
    W[np.ix_(pos, neg)] = 0
    W[np.ix_(neg, pos)] = 0
    D = np.array(np.sqrt(np.sum(W, axis=1))).flatten()  # sqrt of the degrees

    Chis = np.zeros((X.shape[0], 3))
    Chis[pos, 0] = 1.
    Chis[neg, 1] = 1.
    if Ltype == 'normed':
        Chis[:, 0] *= D
        Chis[:, 0] /= sp.linalg.norm(Chis[:, 0])
        Chis[:, 1] *= D
        Chis[:, 1] /= sp.linalg.norm(Chis[:, 1])
        Chis[:, 2] = Chis[:,0] + Chis[:,1]
        Chis[:, 2] /= sp.linalg.norm(Chis[:, 2])
    # u = np.sum(Chis, axis=1) 
    thetas = np.arange(0., 1.1, 0.1)
    u = np.zeros((X.shape[0], len(thetas)))
    for i, theta in enumerate(thetas):
        u[:, i] = theta * Chis[:, 0] + (1-theta) * Chis[:, 1]
        u[:, i] /= sp.linalg.norm(u[:, i])
    return u


def voting_record_test(params, debug=False):
    """
    params:
        knn
        sigma
        zp_k
        Ltype
        n_eigs
        ------------
        n_fid
        tau
        alpha
    """
    if not debug:
        if get_prev_run(
            function    = 'voting_record_test', 
            params      = params, 
            git_commit  = None) is not None:
            print("Found previous voting-record test run")
            return 


    X, labels = load_voting_records()
    gm = Graph_manager()
    graph_params = {
        'knn'   : params['knn'],
        'sigma' : params['sigma'],
        'zp_k'  : params['zp_k'],
        'Ltype' : params['Ltype'],
        'n_eigs': params['n_eigs'],
    }
    # compute or load eigenvectors and eigenvalues
    w, v = gm.from_features(X, graph_params, debug=debug)
    # figure out groundtruth etc of disconnected graph
    chis = compute_groundtruth_u(X, labels, params)
    n_fid = params['n_fid']
    fid = {
        1 : np.where(labels == 1)[0][:n_fid[1]],
        -1: np.where(labels ==-1)[0][:n_fid[-1]]
    }
    lab_ind = list(fid[1])
    lab_ind.extend(list(fid[-1]))
    diag = np.zeros(X.shape[0])
    diag[lab_ind] = 1.
    B = sp.sparse.diags(diag, format='csr')
    # compute statistics
    # load parameters
    alpha = params['alpha']
    T = np.power(0.6, range(5, 20))
    log = {}
    for i in range(chis.shape[1]):
        log['bias_%d'%i] = []
    log['trc'] = []
    log['trcbc'] = []
    for i, tau in enumerate(T):
        gamma = tau**alpha
        d_inv = (tau ** (-2. * alpha)) * np.power(w + tau**2., alpha)
        # prior_inv : C_{tau,eps}^{-1}, where
        # C_{tau, eps}^{-1} = tau^{-2alpha}(L + tau^2 I)^alpha
        prior_inv = v.dot(sp.sparse.diags(d_inv, format='csr').dot(v.T))
        # B/gamma^2
        B_over_gamma2 = B /(gamma**2.)
        # post_inv  : (B/gamma^2 + C_{tau,\eps}^{-1})^{-1}
        post_inv  = prior_inv + B_over_gamma2
        # C^{-1}
        post = sp.linalg.inv(post_inv)
        bias = sp.linalg.norm(post.dot(prior_inv.dot(chis)), axis=0)
        for  ii, b in enumerate(bias):
            log['bias_%d'%ii] +=  [b ** 2]
        # Calculate Tr(C)
        log['trc'] += [sp.trace(post)]
        # Calculate Tr(CBC)/gamma^2
        post2 = post.dot(post)
        trCBC = sp.trace(post2[np.ix_(lab_ind, lab_ind)])
        log['trcbc'] += [trCBC/(gamma**2.)]

    if debug:
        return log
    with mlflow.start_run():
        mlflow.set_tag('function', 'voting_record_test')
        mlflow.log_params(params)
        for i in range(len(T)):
            for ii in range(chis.shape[1]):
                mlflow.log_metric('bias_%d'%ii, log['bias_%d'%ii][i], step=i)
            mlflow.log_metric('trc', log['trc'][i], step=i)
            mlflow.log_metric('trcbc', log['trcbc'][i], step=i)
        return log
        # compute or load data

         



if __name__ == "__main__":
    # mlflow.set_tracking_uri('http://0.0.0.0:8000')
    # mlflow.set_experiment('voting-record')
    params = {
        'knn'      : None,
        'sigma'    : 1.0,
        'zp_k'     : 7,
        'Ltype'    : 'normed',
        'n_eigs'   : None,
        'n_fid'    : {1:5, -1:5},
    }
    ALPHAS = [0.5, 0.75, 1, 1.25, 2, 2.5, 3, 8, 16]
    SIGMAS = [0.5 * i for i in range(1,62)]
    total = len(ALPHAS) * len(SIGMAS)
    i = 0
    for alpha in ALPHAS:
        for sigma in SIGMAS:
            i += 1
            print("{}/{}".format(i, total))
            params['sigma'] = sigma
            params['alpha'] = alpha
            voting_record_test(params, debug=False)

