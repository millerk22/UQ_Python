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

    gm = Graph_manager(N = X.shape[0])
    A  = gm.sqdist(X.T, X.T)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == -1)[0]
    W = gm.compute_similarity_graph(A, knn, sigma) 
    W[np.ix_(pos, neg)] = 0
    W[np.ix_(neg, pos)] = 0
    D = np.array(np.sqrt(np.sum(W, axis=1))).flatten()  # sqrt of the degrees

    Chis = np.zeros((X.shape[0], 2))
    Chis[pos, 0] = 1.
    Chis[neg, 1] = 1.
    if Ltype == 'normed':
        Chis[:, 0] *= D
        Chis[:, 1] *= D
    u = np.sum(Chis, axis=1) 
    u /= sp.linalg.norm(u)
    return u


def voting_record_test(params):
    """
    params:
        knn
        sigma
        Ltype
        n_eigs
        ------------
        n_fid
        tau
        alpha
    """
    if get_prev_run(
        function    = 'voting_record_test', 
        params      = params, 
        git_commit  = None) is not None:
        print("Found previous voting-record test run")
        return 
    with mlflow.start_run():
        mlflow.set_tag('function', 'voting_record_test')
        mlflow.log_params(params)
        # compute or load data
        data_uri = load_voting_records()
        gm = Graph_manager()
        graph_params = {
            'knn'   : params['knn'],
            'sigma' : params['sigma'],
            'Ltype' : params['Ltype'],
            'n_eigs': params['n_eigs'],
            'data_uri' : data_uri
        }
        # compute or load eigenvectors and eigenvalues
        eigs = load_uri(gm.from_features(graph_params), 'eigs.npz')
        w, v = eigs['w'], eigs['v']
        # figure out groundtruth etc of disconnected graph
        data = load_uri(data_uri, 'data.npz')
        X, labels = data['X'], data['labels']
        u = compute_groundtruth_u(X, labels, params)
        num_fid = params['n_fid']
        fid = {
            1 : np.where(labels == 1)[0][:num_fid[1]],
            -1: np.where(labels ==-1)[0][:num_fid[-1]]
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
        for i, tau in enumerate(T):
            gamma = tau**alpha
            d_inv = (tau ** (-2. * alpha)) * np.power(w + tau**2., alpha)
            # prior_inv : C_{tau,eps}^{-1}, where
            # C_{tau, eps}^{-1} = tau^{-2alpha}(L + tau^2 I)^alpha
            prior_inv = v.dot(sp.sparse.diags(d_inv, format='csr').dot(v.T))
            # B/gamma^2
            B_over_gamma2 = B / (gamma**2.)
            # post_inv  : (B/gamma^2 + C_{tau,\eps}^{-1})^{-1}
            post_inv  = prior_inv + B_over_gamma2
            # C^{-1}
            post = sp.linalg.inv(post_inv)
            bias = sp.linalg.norm(post.dot(prior_inv.dot(u)))
            mlflow.log_metric('bias', bias ** 2, step=i)

            # Calculate Tr(C)
            mlflow.log_metric('trc', sp.trace(post), step=i)

            # Calculate Tr(CBC)/gamma^2
            post2 = post.dot(post)
            trCBC = sp.trace(post2[np.ix_(lab_ind, lab_ind)])
            mlflow.log_metric('trcbc', trCBC/(gamma**2.), step=i)


         



if __name__ == "__main__":
    mlflow.set_experiment('voting-record')
    graph_params = {
        'knn'      : None,
        'sigma'    : 1.3,
        'Ltype'    : 'normed',
        'n_eigs'   : None,
        'n_fid'    : {1:5, -1:5},
    }
    ALPHAS = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3.]
    SIGMAS = [0.5 * i for i in range(1,31)] 
    for alpha in ALPHAS:
        for sigma in SIGMAS:
            graph_params['sigma'] = sigma
            graph_params['alpha'] = alpha
            voting_record_test(graph_params)
