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

import gc

import json
import pickle

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import BUCKET_NAME, source_loc, output_loc, config_loc

sys.path.insert(0, source_loc)
from utilities.utility_functions import split_data, select_features, get_bq_data_sample, pickle_and_stream_to_gcs

import warnings
warnings.filterwarnings('ignore')

best_trial_score = -1.0
best_trees_per_fold = [] #empty list to hold # of trees for each fold of best optuna trial

if __name__ == "__main__":
    print("\n--- Training & Tuning LGBM Ranker")

    # ----- READ BIG QUERY DATA INTO DATAFRAME -----
    df_full = get_bq_data_full()
    
    # ----- SELECT FEATURES AND SPLIT DATA -----
    
    #Select only features, sort columns, and target column for training
    df_features, feature_list, sort_list = select_features(df_full)
    
    #Perform split and only return train_val_df to save memory
    train_val_df, _, _ = split_data(df_features)
    train_val_df = train_val_df.reset_index(drop = True)
    
    del df_features
    gc.collect()
    
    # ----- DEFINE BAYESIAN SEARCH OBJECTIVE -----
    
    #Set n_estimators to 2000 for safe cap since early stopping callback is used
    #Set learning rate to 0.05 (half of default for smaller steps while also not going too low due to data size)
    
    #Set objective to lambdarank
    
    #Use ndcg@5 as the evaluation metric to align with the business case. NDCG @5 utilized instead of recall at 5 for base model training because order matters to a degree
    #and should be used to display where each item in the widget is displayed (due to the nature of how the human eye reads)
    
    #Goal of this light hyperparameter tuning is to find optimal shape of trees rather trying to squeeze out fractional gains with a lower learning rate
    
    #reg lambda is looked at for L2 regularization due to few features combining for majority of importance (found in eda feature screening)
    #Note: reg lambda penalizes the model for splitting on dominant features over and over again. It limits the importance of dominant features by preventing trees
    #      from over-relying on these dominant features.
    
    #Avoid max depth and use num leaves due to leaf-wise growth for lgbm (Allow for assymetrical growth across branches)
    
    #Explore feature fraction (create diversity in trees for regularization/generalization)
    
    def objective(trial):
        """
        Optuna objective function for training and evaluating LightGBM ranker hyperparameters
        using stratified group k-fold cross-validation with NDCG@5 optimization.

        Parameters:
        -----------
        trial : optuna.trial.Trial
            A single trial object used for suggesting hyperparameters.

        Returns:
        --------
        float
            The average cross-validation score (NDCG@5) for the trial.
        """
        #Use global best_overall_score to track best score across all trials so best predictions can be saved locally
        global best_trial_score, best_trees_per_fold
        
    
        #Define trial hyperparameters
        params = {
                'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': 5,
                'device': 'gpu', 'learning_rate': 0.05, 'n_estimators': 2000,
                'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True)
                }
    
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
    
            #Because we need to sort by user order groups for the ranker model, keep track of the original index to correctly map predictions back to user orders.
            val_fold['original_index'] = val_fold.index
            
            # Sort after split to ensure user_id, anchor_order_number groups are aligned for lgbm ranker model
            train_fold = train_fold.sort_values(by=['user_id', 'anchor_order_number'])
            val_fold = val_fold.sort_values(by=['user_id', 'anchor_order_number'])
        
            #create df for x_train, x_val, and numpy array for y_train, and y_val (Use arrays to remove index map and reduce memory overhead)
            x_train = train_fold[feature_list]
            y_train = train_fold['label_reordered'].to_numpy(dtype=np.int8)
            x_val = val_fold[feature_list]
            y_val = val_fold['label_reordered'].to_numpy(dtype=np.int8)
    
            print('x_train shape: ', x_train.shape)
            print('y_train shape: ', y_train.shape)
            print('x_val shape: ', x_val.shape)
            print('y_val shape: ', y_val.shape)
        
        
            #Define train and val group size arrays for lgbm ranker
            #Set sort = False to avoid group size misalignment and reduce sorting resource overhead
            
            group_size_train = train_fold.groupby(['user_id', 'anchor_order_number'], sort=False).size().to_numpy()
            group_size_val = val_fold.groupby(['user_id', 'anchor_order_number'], sort=False).size().to_numpy()
        
            print('# of training user groups: ', len(group_size_train))
            print('# of validation user groups: ', len(group_size_val))
    
            #Create lgbm ranker model instance utilize params dictionary for hyperparameters
            lgbm_ranker = lgb.LGBMRanker(**params)
    
            #Create pruning callback to kill poor performing trials early
            pruning_callback = LightGBMPruningCallback(trial, 'ndcg@5')
    
            #Fit the model
            #Utilize early stopping for regularization. Set to 50 rounds to balance with learning rate 0.05 
            
            lgbm_ranker.fit(x_train, y_train, group=group_size_train, eval_set=[(x_val, y_val)], eval_group=[group_size_val],
                            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                            pruning_callback]
            )
            
            # Append validation ndcg@5 to cv scores 
            cv_scores.append(lgbm_ranker.best_score_['valid_0']['ndcg@5'])
    
            #Save predictions for this fold and map to original index. By the end of all folds there will be a prediction for all of train_val_df mapped back to the original index
            trial_oof_preds[val_fold['original_index']] = lgbm_ranker.predict(x_val)
    
            #Append tree count to trial_trees_per_fold
            trial_trees_per_fold.append(lgbm_ranker.best_iteration_) 
    
            # Remove from memory to free up RAM due to data size
            del train_fold, val_fold, x_train, x_val, y_train, y_val, lgbm_ranker
            gc.collect()
    
        avg_cv_score = np.mean(cv_scores)
    
        #If this is the best trial so far, save the oof predictions locally 
        if avg_cv_score > best_trial_score:
            best_trial_score = avg_cv_score
            best_trees_per_fold = trial_trees_per_fold
           
            oof_df = train_val_df[['user_id', 'anchor_order_number', 'label_reordered']].copy()
            oof_df['lgbm_ranker_pred'] = trial_oof_preds
            oof_df.to_parquet(os.path.join(output_loc, "lgbm_ranker_oof_preds.parquet"))
    
        # Remove from memory to free up RAM due to data size
        del trial_oof_preds
        gc.collect() 
                
        return avg_cv_score
    
    # ----- PERFORM BAYESIAN SEARCH -----
    
    #Initialize study. Set it to maximize the objective
    study = optuna.create_study(direction='maximize')
    
    #Optimize utilizing objective function
    #Set trials = 25 to enable the study to wrong long enough to find optimized parameters while also balancing data size
    study.optimize(objective, n_trials=25)
    
    #Get optimal trees per fold
    optimal_trees = int(np.mean(best_trees_per_fold))
    
    #Create params and trees dictionary
    params = {
            "best_params": study.best_params,
            "optimal_trees": optimal_trees,
            "best_trees_per_fold": best_trees_per_fold
            }
    
    # ----- SAVE BEST PARAMETERS LOCALLY -----
    param_file_path = os.path.join(config_loc, "lgbm_ranker_best_params.json")
    with open(param_file_path, "w") as f:
        json.dump(params, f)
    
    # ---- TRAIN MODEL ON FULL DATA USING BEST PARAMS -----
    
    train_full = train_val_df.sort_values(by=['user_id', 'anchor_order_number'])
    x_full = train_full[feature_list].copy()
    y_full = train_full['label_reordered'].to_numpy(dtype=np.int8)
    
    #Get group size array for lgbm_ranker
    group_size_full = train_full.groupby(['user_id', 'anchor_order_number'], sort=False).size().to_numpy()
    
    #Create lgbm ranker instance (use optimal trees and best_params found from study)
    lgbm_ranker_final = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=5, device='gpu', 
        learning_rate=0.05, n_estimators= optimal_trees, **study.best_params)
    
    #Fit the model
    lgbm_ranker_final.fit(x_full, y_full, group=group_size_full)
    
    # ---- SAVE MODEL TO STORAGE BUCKET -----
    
    pickle_and_stream_to_gcs(lgbm_ranker_final, BUCKET_NAME, 'lgbm_ranker_base_model.pkl')
    
    print("\n--- Training Complete. LGBM Ranker model saved to storage.")