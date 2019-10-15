import sys
sys.path.append('..')
from util.mlflow_util import load_uri, get_prev_run
import numpy as np
import scipy.sparse as sps
import scipy.linalg as sla
import os
import mlflow

class Graph_manager(object):
    """
    Instantiated from a Data_obj (features) or generate synthetic data
    Stores relavent parameters of generating similarity graphs from a given
    Data_obj or generating synthetic network

    Necessary Fields:
    self.param

    Optional Fields:
    self.distance_mat
    self.similarity_mat
    self.laplacian
    self.eigenvalues
    self.eigenvectors
    """

    def __init__(self, N=None):
        self.N = N
        return

    def __del__(self):
        try:
            os.remove('./eigs.npz')
        except:
            pass
        return

    def load_data_obj(self, data_obj):
        """
        Load from a Data_obj
        Sets:
            self.data_obj
        """
        self.data_obj = data_obj
        pass

    def update_param(self, new_param):
        """
        new_param : dictionary with param name : param value
        """
        pass
    

    # Needed for all Graph_manager 

    def get_spectrum(self, ks, store=False):
        """
        Returns eigenvalues and eigenvectors specified in ks
        If they are computed before, compute them
        """
        pass

    # Only for Data_obj
    def from_features(self, params, debug=False):
        """
        load from features using params
        params:
            data_uri
            knn
            sigma
            Ltype
            n_eigs
        """
        if not debug:
            prev_run = get_prev_run('Graph_manager.from_features', 
                                    params, None)
            if prev_run is not None:
                print('Found previous eigs')
                return os.path.join(prev_run.info.artifact_uri, 'eigs.npz')


        print('Compute eigs')
        data = load_uri(params['data_uri'])
        self.N = len(data['X'])
        A = self.sqdist(data['X'].T, data['X'].T)
        A = self.compute_similarity_graph(
            distance_mat = A, 
            knn          = params['knn'],
            sigma        = params['sigma'])
        A = self.compute_laplacian(A,
            Ltype = params['Ltype'])
        w, v = self.compute_spectrum(A, n_eigs=params['n_eigs'])

        np.savez('./eigs.npz', w=w, v=v)

        if debug:
            return './eigs.npz'

        with mlflow.start_run(nested=True):
            mlflow.set_tag('function', 'Graph_manager.from_features')
            mlflow.log_params(params)
            mlflow.log_artifact('./eigs.npz')
            return os.path.join(mlflow.get_artifact_uri(), 'eigs.npz')

    def sqdist(self, X, Y):
        """
        Computes dense pairwise euclidean distance between X and Y
        """
        m = X.shape[1]
        n = Y.shape[1]
        Yt = Y.T
        XX = np.sum(X*X, axis=0)
        YY = np.sum(Yt*Yt, axis=1).reshape(n, 1)
        return np.tile(XX, (n, 1)) + np.tile(YY, (1, m)) - 2*Yt.dot(X)

    def compute_similarity_graph(self, distance_mat, knn, sigma):
        """
        Computes similarity graph using parameters specified in self.param 
        """
        # Probably we want to set all default parameters in one place
        if knn is None:
            knn = self.N - 1
        ind_knn = np.argsort(distance_mat, axis=1)[:, 1:knn+1]
        Dknn = distance_mat[(np.arange(self.N).reshape(self.N,1),ind_knn)]
        I = np.tile(np.arange(self.N).reshape(self.N,1), (1,knn)).flatten()
        J = ind_knn.flatten()
        Dknn = np.exp(-Dknn/sigma)
        W = sps.csr_matrix((Dknn.flatten() , (I, J)), shape=(self.N, self.N))
        W = 0.5*(W+W.T)
        return W

    def compute_laplacian(self, W, Ltype):
        """
        Computes the graph Laplacian using parameters specified in self.params
        """
        if Ltype == 'normed':
            L = sps.csgraph.laplacian(W, normed=True)
        else:
            print('not implemented!')
        return L

    def compute_spectrum(self, L, n_eigs):
        """
        Computes first n_eigs smallest eigenvalues and eigenvectors
        """
        if n_eigs is None:
            n_eigs = self.N
        if n_eigs > int(self.N/2):
            w, v = sla.eigh(L.toarray(), eigvals=(0,n_eigs-1))
        else:
            w, v = sps.linalg.eigsh(L, k=n_eigs, which='SM')
        return w, v