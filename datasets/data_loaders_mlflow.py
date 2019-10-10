"""
Experiment with mlflow module
"""
import sys
sys.path.append('..')
import os
import mlflow
import tempfile
import numpy as np
from util.mlflow_util import get_prev_run

def load_voting_records(filepath='./datasets/VOTING-RECORD/'):
    print("Process raw voting records data")
    mlflow.set_experiment('voting-record')
    prev_run = get_prev_run(
        function    = 'data_loaders_mlflow.load_voting_records', 
        params      = {}, 
        git_commit  = None,
        experiment_name = 'voting-record')
    if prev_run is not None:
        print("Found previously processed voting record data")
        data_uri = os.path.join(
            prev_run.info.artifact_uri,
            'data')
        return data_uri
    with mlflow.start_run(nested=True) as mlrun:
        mlflow.set_tag('function', 'data_loaders_mlflow.load_voting_records')
        filename = 'house-votes-84.data'
        with open(filepath + filename, 'r') as f:
            lines = list(f)
            f.close()
        # maps y, n, ? to numerical values
        vote2num = {'y' : 1, 'n' : -1, '?' : 0}
        party2num = {'democrat': 1, 'republican': -1}
        X = [list(map(lambda x: vote2num[x], line.strip().split(',')[1:])) 
            for line in lines]
        X = np.array(X)
        labels = [party2num[line.split(',')[0]] for line in lines]
        labels = np.array(labels)
        data_file = os.path.join(filepath, 'data.npz')
        np.savez(data_file, X=X, labels=labels) 
        mlflow.log_artifact(data_file, 'data')
        os.remove(data_file)
        data_uri = mlflow.get_artifact_uri('data')
        return data_uri

if __name__ == '__main__':
    load_voting_records()
