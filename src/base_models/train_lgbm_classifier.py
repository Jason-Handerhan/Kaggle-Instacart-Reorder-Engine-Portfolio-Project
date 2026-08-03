#Imports and Setups
import sys
import os

import bigframes.pandas as bpd
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import ndcg_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import shuffle

import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb

import mlflow
import mlflow.lightgbm

import gcsfs
import gc

import json
import pickle

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import BUCKET_NAME, source_loc, output_loc, config_loc, MLFLOW_TRACKING_URI, base_model_trials

sys.path.insert(0, source_loc)
from utilities.utility_functions import split_data, select_features, get_bq_data_full, pickle_and_stream_to_gcs

import warnings
warnings.filterwarnings('ignore')

best_trial_score = -1.0
best_trees_per_fold = [] #empty list to hold trees for each fold of best optuna trial

if __name__ == "__main__":
    print("\n--- Training & Tuning LGBM Classifier")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("LGBM_Classifier_Optimization")

    with mlflow.start_run(run_name="LGBM_Classifier_Training_Tuning") as parent_run:
    
        # ----- READ BIG QUERY DATA INTO DATAFRAME -----
        df_full = get_bq_data_full()
        
        # ----- SELECT FEATURES AND SPLIT DATA -----
        
        #Select only features an target column for training
        df_features, feature_list, sort_list = select_features(df_full)
        
        #Perform split and only return train_val_df to save memory
        train_val_df, _, _ = split_data(df_features)
        train_val_df = train_val_df.reset_index(drop = True)
        
        del df_features
        gc.collect()
        
        # ----- DEFINE BAYESIAN SEARCH OBJECTIVE -----
        
        #Set n_estimators to 2000 for safe cap since early stopping callback is used
        #Set learning rate to 0.05 (half of default for smaller steps while also not going too low due to data size)
        #Set objective to binary for binary classification
        
        #Set evaluation metric to average precision (precision-recall AUC) due to class imbalance. ROC AUC is avoided due to the potential inflation
        #of this metric due to true negatives. With average precision, True negatives are ignored and the focus is on minimizing false negatives and false positives.
        #This aligns with the business case as we want to limit false negatives (predict re-orders correctly) while also minimizing the 
        #amount of false negatives (predicting a reorder that was not re-ordered).
        
        #Goal of this light hyperparameter tuning is to find optimal shape of trees rather trying to squeeze out fractional gains with a lower learning rate
        
        #reg lambda is looked at for L2 regularization due to few features combining for majority of importance (found in eda feature screening)
        #Note: reg lambda penalizes the model for splitting on dominant features over and over again. It limits the importance of dominant features by preventing trees
        #      from over-relying on these dominant features.
        
        #Avoid max depth and use num leaves due to leaf-wise growth for lgbm (Allow for assymetrical growth across branches)
        
        #Explore feature fraction (create diversity in trees for regularization/generalization)
        
        #Set min_split_gain = 0.1. This acts a slight method of regularization preventing the model from overfitting
        # by making splits on microscopic gains
        
        def objective(trial):
            """
            Optuna objective function for training and evaluating LightGBM classifier hyperparameters
            using stratified group k-fold cross-validation.

            Parameters:
            -----------
            trial : optuna.trial.Trial
                A single trial object used for suggesting hyperparameters.

            Returns:
            --------
            float
                The average cross-validation score (average precision) for the trial.
            """
            global best_trial_score, best_trees_per_fold
        
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):

                #Define trial hyperparameters
                params = {
                        'objective': 'binary', 'metric': 'average_precision',
                        'device': 'cpu', 'learning_rate': 0.05, 'n_estimators': 2000,
                        'min_split_gain': 0.1,
                        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True)
                        }
                
                mlflow.log_params(params)

                sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
            
                #Create empty cv cores list to track cross validation scores
                cv_scores = []
            
                #Create array to track predictions
                trial_oof_preds = np.zeros(len(train_val_df))
            
                #Create list to track trees per fold
                trial_trees_per_fold = []
                    
                for train_index, val_index in sgkf.split(train_val_df[feature_list], train_val_df['label_reordered'], groups=train_val_df['user_id']):
                
                    print('Starting new fold...')
            
                    #Create train and val data sets for current fold
                    train_fold = train_val_df.iloc[train_index].copy()
                    val_fold = train_val_df.iloc[val_index].copy()
            
                    #Use the same method used for the rankers to track the original index. This ensures code consistency and acts as an insurance policy against index scrambling.
                    val_fold['original_index'] = val_fold.index
                
                    #create df for x_train, x_val, and numpy array for y_train, and y_val (Use arrays to remove index map and reduce memory overhead)
                    x_train = train_fold[feature_list]
                    y_train = train_fold['label_reordered'].to_numpy(dtype=np.int8)
                    x_val = val_fold[feature_list]
                    y_val = val_fold['label_reordered'].to_numpy(dtype=np.int8)
            
                    print('x_train shape: ', x_train.shape)
                    print('y_train shape: ', y_train.shape)
                    print('x_val shape: ', x_val.shape)
                    print('y_val shape: ', y_val.shape)
            
                    #Create lgbm classifier model instance utilize params dictionary for hyperparameters
                    lgbm_classifier = lgb.LGBMClassifier(**params)
            
                    #Create pruning callback to kill poor performing trials early
                    pruning_callback = LightGBMPruningCallback(trial, 'average_precision')
            
                    #Fit the model
                    #Utilize early stopping for regularization. Set to 50 rounds to balance with learning rate 0.05 
                    
                    lgbm_classifier.fit(x_train, y_train, eval_set=[(x_val, y_val)],
                                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                                    pruning_callback]
                    )
                    
                    # Append validation ndcg@5 to cv scores 
                    cv_scores.append(lgbm_classifier.best_score_['valid_0']['average_precision'])
            
                    #Save predictions for this fold and map to original index. By the end of all folds there will be a prediction for all of train_val_df mapped back to the original index
                    trial_oof_preds[val_fold['original_index']] = lgbm_classifier.predict_proba(x_val)[:, 1]
            
                    #Append tree count to trial_trees_per_fold
                    trial_trees_per_fold.append(lgbm_classifier.best_iteration_) 
            
                    # Remove from memory to free up RAM due to data size
                    del train_fold, val_fold, x_train, x_val, y_train, y_val, lgbm_classifier
                    gc.collect()
            
                avg_cv_score = np.mean(cv_scores)
                mlflow.log_metric("avg_cv_average_precision", avg_cv_score)
            
                #If this is the best trial so far, save the oof predictions locally 
                if avg_cv_score > best_trial_score:
                    best_trial_score = avg_cv_score
                    best_trees_per_fold = trial_trees_per_fold
                    
                    oof_df = train_val_df[['user_id', 'anchor_order_number', 'label_reordered']].copy()
                    oof_df['lgbm_classifier_pred'] = trial_oof_preds
                    oof_df.to_parquet(os.path.join(output_loc, "lgbm_classifier_oof_preds.parquet"))
            
                # Remove from memory to free up RAM due to data size
                del trial_oof_preds
                gc.collect() 
                        
                return avg_cv_score
        
        # ----- PERFORM BAYESIAN SEARCH -----
        
        #Initialize study. Set it to maximize the objective
        study = optuna.create_study(direction='maximize')
        
        #Optimize utilizing objective function
        #Set trials = 25 to enable the study to wrong long enough to find optimized parameters while also balancing data size
        study.optimize(objective, n_trials=base_model_trials)
        
        #Get optimal trees per fold
        optimal_trees = int(np.mean(best_trees_per_fold))
        
        #Create params and trees dictionary
        params = {
                "best_params": study.best_params,
                "optimal_trees": optimal_trees,
                "best_trees_per_fold": best_trees_per_fold
                }
        
        # ----- SAVE BEST PARAMETERS LOCALLY -----
        param_file_path = os.path.join(config_loc, "lgbm_classifier_best_params.json")
        with open(param_file_path, "w") as f:
            json.dump(params, f)
        
        # ---- TRAIN MODEL ON FULL DATA USING BEST PARAMS -----
        
        train_full = train_val_df
        x_full = train_full[feature_list].copy()
        y_full = train_full['label_reordered'].to_numpy(dtype=np.int8)
        
        #Create lgbm classifier instance (use optimal trees and best_params found from study)
        lgbm_classifier_final = lgb.LGBMClassifier(objective='binary', metric='average_precision', device='cpu', learning_rate=0.05, n_estimators=optimal_trees, **study.best_params)
        
        #Fit the model
        lgbm_classifier_final.fit(x_full, y_full)
        
        # ---- SAVE MODEL TO STORAGE BUCKET -----
        
        pickle_and_stream_to_gcs(lgbm_classifier_final, BUCKET_NAME, 'lgbm_classifier_base_model.pkl')
        
        # ---- LOG MASTER METRICS & ARTIFACTS TO MLFLOW -----
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_average_precision", study.best_value)
        mlflow.log_metric("optimal_trees", optimal_trees)
        mlflow.log_artifact(param_file_path)
        
        oof_file_path = os.path.join(output_loc, "lgbm_classifier_oof_preds.parquet")
        if os.path.exists(oof_file_path):
            mlflow.log_artifact(oof_file_path)
            
        mlflow.lightgbm.log_model(lgbm_classifier_final, artifact_path="lgbm_classifier_base_model")
        
        print("\n--- Training Complete. LGBM Classifier model saved to storage.")