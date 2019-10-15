export MLFLOW_TRACKING_URI='http://0.0.0.0:8000'
pwd=`pwd`
mkdir -p mlruns-db
mlflow server --backend-store-uri postgresql://mlflow:mlflow@localhost/mlflow --default-artifact-root file:$pwd/mlruns-db -h 0.0.0.0 -p 8000
