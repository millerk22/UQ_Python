#from mcmc_classes import *
from datasets.data_loaders_mlflow import load_voting_records
from util.mlflow_util import get_prev_run
import mlflow
import numpy as np
from datasets.Graph_manager import Graph_manager

def voting_record_test(params):
    mlflow.set_experiment('voting-record')
    """
    if get_prev_run(
        function    = 'voting_record_test', 
        params      = params, 
        git_commit  = None, 
        experiment_name = 'voting-record') is not None:
        print("Found previous voting-record test run")
        return 
    """
    with mlflow.start_run() as mlrun:
        mlflow.set_tag('function', 'voting_record_test')
        data_uri = load_voting_records()
        graph_params['data_uri'] = data_uri
        gm = Graph_manager('voting-record')
        print(gm.from_features(graph_params))


if __name__ == "__main__":
    graph_params = {
        'knn'      : None,
        'sigma'    : 1.3,
        'Ltype'    : 'normed',
        'n_eigs'   : None
    }
    voting_record_test(graph_params)
