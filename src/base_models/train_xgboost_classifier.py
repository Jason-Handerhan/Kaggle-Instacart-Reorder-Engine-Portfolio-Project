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
from optuna.integration import XGBoostPruningCallback
import xgboost as xgb 

import mlflow
import mlflow.xgboost

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
best_trees_per_fold = [] #empty list to hold # of trees for each fold of best optuna trial

if __name__ == "__main__":
    print("\n--- Training & Tuning XGBOOST Classifier")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("XGBoost_Classifier_Optimization")

    with mlflow.start_run(run_name="XGBoost_Classifier_Training_Tuning") as parent_run:
    
        # ----- READ BIG QUERY DATA INTO DATAFRAME -----
        df_full = get_bq_data_full()
        
        # ----- SELECT FEATURES AND SPLIT DATA -----
        
        #Select only features, sort columns, and target column for training
        df_features, feature_list, sort_list = select_features(df_full)
        
        #Perform split and only return train_val_df to save memory
        train_val_df, _, _ = split_data(df_features)
        train_val_df = train_val_df.reset_index(drop = True)
        
        #Remove df_features from memory
        del df_features
        gc.collect()
        
        # ----- DEFINE BAYESIAN SEARCH OBJECTIVE -----
        
        #Set n_estimators to 2000 for safe cap
        #Set learning rate to 0.05 (half of default for smaller steps while also not going too low due to data size)
        
        #Set the objective to binary:logistic for binary classificaiton
        
        #Set evaluation metric to average precision (precision-recall AUC) due to class imbalance. ROC AUC is avoided due to the potential inflation
        #of this metric due to true negatives. With average precision, True negatives are ignored and the focus is on minimizing false negatives and false positives.
        #This aligns with the business case as we want to limit false negatives (predict re-orders correctly) while also minimizing the 
        #amount of false negatives (predicting a reorder that was not re-ordered).
        
        #Goal of this light hyperparameter tuning is to find optimal shape of trees rather trying to squeeze out fractional gains with a lower learning rate.
        
        #reg lambda is looked at for L2 regularization due to few features combining for majority of importance (found in eda feature screening)
        #Note: reg lambda penalizes the model for splitting on dominant features over and over again. It limits the importance of dominant features by preventing trees
        #      from over-relying on these dominant features.
        
        #Explore max depth in bayesian search instead of num_leaves (due to level_wise growth for xgboost instead of leafwise)
        # Limit tree depth to 10 max due to data size and to prevent overfitting (avoid repetitive splitting on same features with only 14 features)
        
        #Explore colsample_bytree (create diversity in trees for regularization/generalization)
        
        #Utilize tree_method: 'hist' for performance due to data size. This utilizes histogram based binning to build decision tress instead of exact greedy splits.
        #Thus instead of evaluating the gain of making a split at every single unique value for every feature at every node, the values for each feature are divided into bins
        #and splits are evaluated only at the bin boundaries (which is significantly less computationally expensive)
        
        #Set gamma = 0.1. This acts a slight method of regularization preventing the model from overfitting by making splits on microscopic gains
        
        def objective(trial):
            """
            Optuna objective function for training and evaluating XGBoost classifier hyperparameters
            using stratified group k-fold cross-validation with average precision optimization.

            Parameters:
            -----------
            trial : optuna.trial.Trial
                A single trial object used for suggesting hyperparameters.

            Returns:
            --------
            float
                The average cross-validation score (average precision / PR-AUC) for the trial.
            """
            #Use global best_overall_score to track best score across all trials so best predictions can be saved locally
            global best_trial_score, best_trees_per_fold
            
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):

                #Define trial hyperparameters
                params = {
                        'objective': 'binary:logistic', 'eval_metric': 'aucpr', 
                        'tree_method': 'hist', 'device': 'cuda', 'learning_rate': 0.05, 'n_estimators': 2000,
                        'gamma': 0.1,
                        'max_depth': trial.suggest_int('max_depth', 5, 10),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True)
                        }
            
                mlflow.log_params(params)

                #Create StratifiedGroupKFold instance
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
            
            
                    #Create pruning callback to kill poor performing trials early
                    pruning_callback = XGBoostPruningCallback(trial, 'validation_0-aucpr')
            
                    #Create xgboost_classifier model instance & utilize params dictionary for hyperparameters
                    #Utilize early stopping for regularization. Set to 50 rounds to balance with learning rate 0.05
                    xgboost_classifier = xgb.XGBClassifier(early_stopping_rounds=50, callbacks=[pruning_callback], **params)
            
                    #Fit the model
                    #verbose = False to prevent flooding of print statements
                    xgboost_classifier.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
                    
                    # Append validation precision recall auc to cv scores 
                    cv_scores.append(max(xgboost_classifier.evals_result()['validation_0']['aucpr']))
            
                    #Save predictions for this fold and map to original index. By the end of all folds there will be a prediction for all of train_val_df mapped back to the original index
                    trial_oof_preds[val_fold['original_index']] = xgboost_classifier.predict_proba(x_val)[:, 1]
            
                    #Append tree count to trial_trees_per_fold
                    trial_trees_per_fold.append(xgboost_classifier.best_iteration) 
            
                    # Remove from memory to free up RAM due to data size
                    del train_fold, val_fold, x_train, x_val, y_train, y_val, xgboost_classifier
                    gc.collect()
            
                avg_cv_score = np.mean(cv_scores)
                mlflow.log_metric("avg_cv_aucpr", avg_cv_score)
            
                #Manual pruning to kill poor performing trails early
                #trial.report(avg_cv_score, step=trial.number)
                #if trial.should_prune():
                 #   raise optuna.exceptions.TrialPruned()
            
                #If this is the best trial so far, save the oof predictions locally 
                if avg_cv_score > best_trial_score:
                    best_trial_score = avg_cv_score
                    best_trees_per_fold = trial_trees_per_fold
                   
                    oof_df = train_val_df[['user_id', 'anchor_order_number', 'label_reordered']].copy()
                    oof_df['xgboost_classifier_pred'] = trial_oof_preds
                    oof_df.to_parquet(os.path.join(output_loc, "xgboost_classifier_oof_preds.parquet"))
            
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
        
        param_file_path = os.path.join(config_loc, "xgboost_classifier_best_params.json")
        with open(param_file_path, "w") as f:
            json.dump(params, f)
        
        # ---- TRAIN MODEL ON FULL DATA USING BEST PARAMS -----
        
        train_full = train_val_df
        x_full = train_full[feature_list].copy()
        y_full = train_full['label_reordered'].to_numpy(dtype=np.int8)
        
        #Create xgboost classifier instance (use optimal trees and best_params found from study)
        xgboost_classifier_final = xgb.XGBClassifier(objective='binary:logistic', eval_metric='aucpr', tree_method='hist', device='cuda', learning_rate=0.05, n_estimators=optimal_trees, **study.best_params)
        
        #Fit the model
        xgboost_classifier_final.fit(x_full, y_full)
        
        # ---- SAVE MODEL TO STORAGE BUCKET -----

        pickle_and_stream_to_gcs(xgboost_classifier_final, BUCKET_NAME, 'xgboost_classifier_base_model.pkl')
        
        # ---- LOG MASTER METRICS & ARTIFACTS TO MLFLOW -----
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_aucpr", study.best_value)
        mlflow.log_metric("optimal_trees", optimal_trees)
        mlflow.log_artifact(param_file_path)
        
        oof_file_path = os.path.join(output_loc, "xgboost_classifier_oof_preds.parquet")
        if os.path.exists(oof_file_path):
            mlflow.log_artifact(oof_file_path)
            
        mlflow.xgboost.log_model(xgboost_classifier_final, artifact_path="xgboost_classifier_base_model")
        
        print("\n--- Training Complete. XGBOOST Classifier model saved to storage.")