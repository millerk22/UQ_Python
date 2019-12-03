from datasets.data_loaders_mlflow import load_MNIST
from datasets.Graph_manager import Graph_manager

if __name__ == '__main__':
    digits = [1,4,7,9]
    num_points = [500] * 4
    seed = 42
    data_params = {
        'digits' : digits,
        'num_points' : num_points,
        'seed' : seed
    }
    X, labels = load_MNIST(data_params)
    gm = Graph_manager()
    graph_params = {
        'knn': 15,
        'Ltype': 'normed',
        'sigma': 1,
        'zp_k' : 7,
        'n_eigs': 100
    }
    w, v = gm.from_features(X, graph_params, debug=True)
    print(w.shape, v.shape)