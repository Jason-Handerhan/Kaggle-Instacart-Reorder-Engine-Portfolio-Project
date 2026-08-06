#Root File Location
root_loc = '/home/jupyter/Instacart_Reorder_Engine'

#Source File Location
source_loc = '/home/jupyter/Instacart_Reorder_Engine/src'

#Local Outputs Location
output_loc = '/home/jupyter/Instacart_Reorder_Engine/outputs'

#Local config location (saving optimal weights and hyperparameters)
config_loc = '/home/jupyter/Instacart_Reorder_Engine/config'

#Google Cloud Storage Bucket
BUCKET_NAME = "instacart-market-basket-analysis-jhanderhan"
BUCKET_PATH = "gs://instacart-market-basket-analysis-jhanderhan"

#Big Query Project ID
project_id = "instacart-ml-model"

#Big Query Results ML Model Results Dataset (Dataset for Saving model evaluation notebook results back to Big Query)
results_dataset = 'ml_model_results'

#Big Query Location
big_query_loc = 'US'

#Python Environment
PY_ENV = "/opt/micromamba/envs/jupyterlab/bin/python3"

#mlflow tracking uri
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

#Base Model Bayesian Search Trials
base_model_trials = 25

#Optuna Ensemble Weight Optimization Trials
ensemble_trials = 30