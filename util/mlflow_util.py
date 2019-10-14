import mlflow
import os
import numpy as np

def get_prev_run(function, params, git_commit):
    query = 'attributes.status = "FINISHED"'
    query += ' and tags."function" = "{}"'.format(function)
    for key, val in params.items():
        query += ' and '
        query += 'params.{} = "{}"'.format(key, val) 
    print(query)
    runs = mlflow.search_runs(filter_string=query)
    if runs.empty:
        return None
    else:
        # TODO should check git_commit
        return mlflow.tracking.MlflowClient().get_run(
            runs.iloc[0].loc['run_id'])

def load_uri(uri, filename, offset=7):
    filepath = os.path.join(uri[offset:], filename)
    return np.load(filepath)