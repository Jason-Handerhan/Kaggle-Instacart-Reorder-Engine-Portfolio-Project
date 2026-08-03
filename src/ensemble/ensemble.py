#Imports and setups
import sys
import os

import pandas as pd
import numpy as np
import pickle
import json
import optuna

import mlflow

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import BUCKET_NAME, source_loc, output_loc, config_loc, MLFLOW_TRACKING_URI, ensemble_trials

sys.path.insert(0, source_loc)
from utilities.utility_functions import compute_rrf_scores, calculate_recall_at_k, evaluate_ensemble

if __name__ == "__main__":
    print("\n--- Tuning Ensemble Weights")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Ensemble_Weight_Optimization")

    with mlflow.start_run(run_name="Ensemble_Weight_Tuning") as parent_run:

        #Create Empty Dictionary to save best weights
        best_weights = {}

        # ----- READ BASE MODEL PREDICTIONS -----

        try:
            lr_preds = pd.read_parquet(os.path.join(output_loc, "lgbm_ranker_oof_preds.parquet"))
            lc_preds = pd.read_parquet(os.path.join(output_loc, "lgbm_classifier_oof_preds.parquet"))
            xr_preds = pd.read_parquet(os.path.join(output_loc, "xgboost_ranker_oof_preds.parquet"))
            xc_preds = pd.read_parquet(os.path.join(output_loc, "xgboost_classifier_oof_preds.parquet"))
            
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("Please ensure base model scripts ran successfully and saved outputs to base_models/base_model_preds_params/ ")
            exit(1)

        # ----- COMPUTE RANK RECIPROCAL FUSION (RRF) SCORES -----

        #Create all_preds df
        base_model_preds = lr_preds[['user_id', 'anchor_order_number', 'label_reordered', 'lgbm_ranker_pred']].copy()
        base_model_preds = base_model_preds.rename(columns= {'lgbm_ranker_pred' : 'lgbm_ranker'})
        base_model_preds['lgbm_classifier'] = lc_preds['lgbm_classifier_pred']
        base_model_preds['xgb_ranker'] = xr_preds['xgboost_ranker_pred']
        base_model_preds['xgb_classifier'] = xc_preds['xgboost_classifier_pred']

        #Compute rrf scores for each base model
        base_model_preds = compute_rrf_scores(base_model_preds, ['lgbm_ranker', 'lgbm_classifier', 'xgb_ranker', 'xgb_classifier'],
                                              ['user_id', 'anchor_order_number'])

        # ----- CREATE OPTUNA OPTIMIZATION OBJECTIVES -----

        #lgbm ranker + xgboost classifier weight tuning
        def obj_ens1(trial):
            """
            Objective function for Optuna to optimize ensemble weights of the 
            LightGBM Ranker and XGBoost Classifier based on NDCG@5.

            Args:
                trial (optuna.trial.Trial): A trial object for hyperparameter optimization.

            Returns:
                float: The NDCG@5 score for the ensembled predictions.
            """
            with mlflow.start_run(run_name=f"ens1_trial_{trial.number}", nested=True):
                w_lr = trial.suggest_float('w_lr', 0.0, 1.0)
                w_xc = trial.suggest_float('w_xc', 0.0, 1.0)

                ensemble_preds = (w_lr/(w_lr+w_xc))*base_model_preds['lgbm_ranker_rrf'] + (w_xc/(w_lr+w_xc))*base_model_preds['xgb_classifier_rrf']
                
                ndcg_at_5 = evaluate_ensemble(ensemble_preds, base_model_preds)
                
                mlflow.log_params({'w_lr': w_lr, 'w_xc': w_xc})
                mlflow.log_metric("ndcg_at_5", ndcg_at_5)
                
                return ndcg_at_5

        #lgbm ranker + lgbm classifer weight tuning
        def obj_ens2(trial):
            """
            Objective function for Optuna to optimize ensemble weights of the 
            LightGBM Ranker and LightGBM Classifier based on NDCG@5.

            Args:
                trial (optuna.trial.Trial): A trial object for hyperparameter optimization.

            Returns:
                float: The NDCG@5 score for the ensembled predictions.
            """
            with mlflow.start_run(run_name=f"ens2_trial_{trial.number}", nested=True):
                w_lr = trial.suggest_float('w_lr', 0.0, 1.0)
                w_lc = trial.suggest_float('w_lc', 0.0, 1.0)
                
                ensemble_preds = (w_lr/(w_lr+w_lc))*base_model_preds['lgbm_ranker_rrf'] + (w_lc/(w_lr+w_lc))*base_model_preds['lgbm_classifier_rrf']

                ndcg_at_5 = evaluate_ensemble(ensemble_preds, base_model_preds)
                
                mlflow.log_params({'w_lr': w_lr, 'w_lc': w_lc})
                mlflow.log_metric("ndcg_at_5", ndcg_at_5)
                
                return ndcg_at_5

        #xgboost ranker + lgbm classifer weight tuning
        def obj_ens3(trial):
            """
            Objective function for Optuna to optimize ensemble weights of the 
            XGBoost Ranker and LightGBM Classifier based on NDCG@5.

            Args:
                trial (optuna.trial.Trial): A trial object for hyperparameter optimization.

            Returns:
                float: The NDCG@5 score for the ensembled predictions.
            """
            with mlflow.start_run(run_name=f"ens3_trial_{trial.number}", nested=True):
                w_xr = trial.suggest_float('w_xr', 0.0, 1.0)
                w_lc = trial.suggest_float('w_lc', 0.0, 1.0)
                
                ensemble_preds = (w_xr/(w_xr+w_lc))*base_model_preds['xgb_ranker_rrf'] + (w_lc/(w_xr+w_lc))*base_model_preds['lgbm_classifier_rrf']

                ndcg_at_5 = evaluate_ensemble(ensemble_preds, base_model_preds)
                
                mlflow.log_params({'w_xr': w_xr, 'w_lc': w_lc})
                mlflow.log_metric("ndcg_at_5", ndcg_at_5)
                
                return ndcg_at_5

        #xgboost ranker + xgboost classifer weight tuning
        def obj_ens4(trial):
            """
            Objective function for Optuna to optimize ensemble weights of the 
            XGBoost Ranker and XGBoost Classifier based on NDCG@5.

            Args:
                trial (optuna.trial.Trial): A trial object for hyperparameter optimization.

            Returns:
                float: The NDCG@5 score for the ensembled predictions.
            """
            with mlflow.start_run(run_name=f"ens4_trial_{trial.number}", nested=True):
                w_xr = trial.suggest_float('w_xr', 0.0, 1.0)
                w_xc = trial.suggest_float('w_xc', 0.0, 1.0)
                
                ensemble_preds = (w_xr/(w_xr+w_xc))*base_model_preds['xgb_ranker_rrf'] + (w_xc/(w_xr+w_xc))*base_model_preds['xgb_classifier_rrf']

                ndcg_at_5 = evaluate_ensemble(ensemble_preds, base_model_preds)
                
                mlflow.log_params({'w_xr': w_xr, 'w_xc': w_xc})
                mlflow.log_metric("ndcg_at_5", ndcg_at_5)
                
                return ndcg_at_5

        #All base model ensemble
        def obj_ens5(trial):
            """
            Objective function for Optuna to optimize ensemble weights of all four 
            base models (LightGBM Ranker & Classifier, XGBoost Ranker & Classifier) 
            based on NDCG@5.

            Args:
                trial (optuna.trial.Trial): A trial object for hyperparameter optimization.

            Returns:
                float: The NDCG@5 score for the ensembled predictions.
            """
            with mlflow.start_run(run_name=f"ens5_trial_{trial.number}", nested=True):
                w_lr = trial.suggest_float('w_lr', 0.0, 1.0)
                w_lc = trial.suggest_float('w_lc', 0.0, 1.0)
                w_xr = trial.suggest_float('w_xr', 0.0, 1.0)
                w_xc = trial.suggest_float('w_xc', 0.0, 1.0)
                tot = w_lr + w_lc + w_xr + w_xc
                
                ensemble_preds = ((w_lr/tot)*base_model_preds['lgbm_ranker_rrf'] + (w_lc/tot)*base_model_preds['lgbm_classifier_rrf'] +
                             (w_xr/tot)*base_model_preds['xgb_ranker_rrf'] + (w_xc/tot)*base_model_preds['xgb_classifier_rrf'])

                ndcg_at_5 = evaluate_ensemble(ensemble_preds, base_model_preds)
                
                mlflow.log_params({'w_lr': w_lr, 'w_lc': w_lc, 'w_xr': w_xr, 'w_xc': w_xc})
                mlflow.log_metric("ndcg_at_5", ndcg_at_5)
                
                return ndcg_at_5

        # ----- FIND OPTIMAL ENSEMBLE WEIGHTS -----

        #lgbm ranker + xgboost classifier weight tuning
        study1 = optuna.create_study(direction='maximize')
        study1.optimize(obj_ens1, n_trials=ensemble_trials)
        best_weights['ens1_lgb_rank_xgb_class'] = study1.best_params

        #lgbm ranker + lgbm classifer weight tuning
        study2 = optuna.create_study(direction='maximize')
        study2.optimize(obj_ens2, n_trials=ensemble_trials)
        best_weights['ens2_lgb_rank_lgb_class'] = study2.best_params

        #xgboost ranker + lgbm classifer weight tuning
        study3 = optuna.create_study(direction='maximize')
        study3.optimize(obj_ens3, n_trials=ensemble_trials)
        best_weights['ens3_xgb_rank_lgb_class'] = study3.best_params

        #xgboost ranker + xgboost classifer weight tuning
        study4 = optuna.create_study(direction='maximize')
        study4.optimize(obj_ens4, n_trials=ensemble_trials)
        best_weights['ens4_xgb_rank_xgb_class'] = study4.best_params

        #All base model ensemble
        study5 = optuna.create_study(direction='maximize')
        study5.optimize(obj_ens5, n_trials=ensemble_trials)
        best_weights['ens5_all_base_models'] = study5.best_params

        # ----- SAVE OPTIMAL ENSEMBLE WEIGHTS -----

        best_weights_path = os.path.join(config_loc, "optimal_ensemble_weights.json")
        with open(best_weights_path, "w") as f:
            json.dump(best_weights, f, indent=4)

        # ---- LOG MASTER METRICS & ARTIFACTS TO MLFLOW -----
        mlflow.log_metric("ens1_best_ndcg5", study1.best_value)
        mlflow.log_metric("ens2_best_ndcg5", study2.best_value)
        mlflow.log_metric("ens3_best_ndcg5", study3.best_value)
        mlflow.log_metric("ens4_best_ndcg5", study4.best_value)
        mlflow.log_metric("ens5_best_ndcg5", study5.best_value)
        
        mlflow.log_artifact(best_weights_path)

        print("\n--- Tuning Complete. Ensemble weights saved to config and logged to MLflow.")