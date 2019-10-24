from datasets.data_loaders_mlflow import load_MNIST
from datasets.Graph_manager import Graph_manager

if __name__ == '__main__':
    digits = [1,4,7,9]
    num_digits = [500] * 4
    seed = 42
    data_params = {
        'digits' : digits,
        'num_digits' : num_digits,
        'seed' : seed
    }
    X, labels = load_MNIST(data_params)
    gm = Graph_manager()
    graph_params = {
        'X' : X,
        'knn': None,
        'Ltype': 'normed',
        'sigma': 1,
        'n_eigs': 100
    }
    w, v = gm.from_features(graph_params)