import pandas as pd
import numpy as np
import bigframes.pandas as bpd
from sklearn.utils import shuffle
from sklearn.metrics import ndcg_score

def split_data(df):
    """
    Split data into Train/Val (80%), Val (10%), and Test (10%).
    Sorts unique users first to guarantee identical splits across multiple scripts.

    Parameters:
    -----------
    df : pd.DataFrame
        The full input dataframe containing a 'user_id' column.

    Returns:
    --------
    tuple of pd.DataFrame
        Returns (train_val_df, val_df, test_df).
    """
    
    # Sort unique users to ensure same splits across multiple scripts
    unique_users = pd.Series(np.sort(df['user_id'].unique()))
    shuffled_users = shuffle(unique_users, random_state = 42).reset_index(drop = True)
    
    n_users = len(shuffled_users)
    train_val_users = shuffled_users[:int(0.8 * n_users)]
    val_users = shuffled_users[int(0.8 * n_users):int(0.9 * n_users)]
    test_users = shuffled_users[int(0.9 * n_users):]
    
    train_val_df = df[df['user_id'].isin(train_val_users)].copy()
    val_df = df[df['user_id'].isin(val_users)].copy()
    test_df = df[df['user_id'].isin(test_users)].copy()
    
    return train_val_df, val_df, test_df


def select_features(df):
    """
    Select features, target, and sort list (for ranking models) for model training.

    Parameters:
    -----------
    df : pd.DataFrame
        The full dataset containing all raw features and target columns.

    Returns:
    --------
    tuple
        Returns (df, feature_list, sort_list) with filtered columns.
    """
    
    feature_list = ['user_prod_avg_days_between_purchases',
                    'user_product_consecutive_order_skips',
                    'user_product_order_penetration_ratio',
                    'user_product_reorder_concentration',
                    'label_user_product_days_since_last_purchased',
                    'user_product_last3order_streak',
                    'user_product_cumulative_purchases',
                    'user_product_purchase_concentration',
                    'user_product_nearing_reorder_ratio',
                    'user_reorder_ratio',
                    'user_product_cart_position_last_order',
                    'user_product_avg_cart_position',
                    'user_distinct_product_count',
                    'user_deptartment_share',
                    'user_aisle_share']
    
    sort_list = ['user_id', 'anchor_order_number']
    
    target_col = ['label_reordered']
    
    keep_list = feature_list + sort_list + target_col
    
    df = df[keep_list]
    

    return df, feature_list, sort_list

def get_bq_data_sample():
    """
    Read big query sql pipeline features and target table into bigframe, 
    select a 10% user sample, and return a pandas dataframe.

    Returns:
    --------
    pd.DataFrame
        A 10% sampled pandas dataframe of the feature table.
    """
    
    # 1. Initialize BigQuery connection strings
    bpd.options.bigquery.project = "instacart-ml-model"
    bpd.options.bigquery.location = "US"
    
    # 2. Load data into BigFrame
    bf_full = bpd.read_gbq("instacart-ml-model.ml_model_feature_table.final_ml_features_table")
    
    # 3. Get unique users
    unique_users = bf_full[['user_id']].drop_duplicates()
    
    # 4. Obtain reproducible 10% sample
    sample_users = unique_users.sample(frac=0.10, random_state=42)
    
    # 5. Merge to bf_full to create 10% sample and convert to df
    df_sample = bf_full.merge(sample_users, on='user_id', how='inner').to_pandas()
    
    
    print(f"Full data shape: {bf_full.shape}")
    print(f"Sample data shape: {df_sample.shape}")

    return df_sample

def get_bq_data_full():
    """
    Read big query sql pipeline features and target table into bigframe, 
    select a 50% user sample, and return a pandas dataframe.

    Returns:
    --------
    pd.DataFrame
        A 50% sampled pandas dataframe of the full feature table.
    """
    
    # 1. Initialize BigQuery connection strings
    bpd.options.bigquery.project = "instacart-ml-model"
    bpd.options.bigquery.location = "US"
    
    # 2. Load data into BigFrame
    bf_full = bpd.read_gbq("instacart-ml-model.ml_model_feature_table.final_ml_features_table")
    
    # 3. Get unique users
    unique_users = bf_full[['user_id']].drop_duplicates()
    
    # 4. Obtain reproducible 50% sample
    sample_users = unique_users.sample(frac=0.50, random_state=42)
    
    # 5. Merge to bf_full to create 50% sample and convert to df
    df_full = bf_full.merge(sample_users, on='user_id', how='inner').to_pandas()
    
    print(f"Full data shape: {df_full.shape}")

    return df_full


def compute_rrf_scores(df, models, group_cols, c=60):
    """
    Converts raw predictions into Reciprocal Rank Fusion scores.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing raw predictions.
    models : list of str
        List of model prediction column names to convert.
    group_cols : list of str
        Columns defining the grouping level (e.g., user and order).
    c : int, default=60
        RRF constant parameter.

    Returns:
    --------
    pd.DataFrame
        Dataframe containing the original data plus new RRF score columns.
    """
    df_rrf = df.copy()
    for col in models:
        rank_series = df_rrf.groupby(group_cols)[col].rank(ascending=False, method='min')
        df_rrf[f'{col}_rrf'] = 1.0 / (c + rank_series)
    return df_rrf



def evaluate_ensemble(ensemble_preds, base_model_preds):
    """
    Evaluate ndcg at k for blended ensemble model.

    Parameters:
    -----------
    ensemble_preds : array-like
        Blended predictions array.
    base_model_preds : pd.DataFrame
        Dataframe containing identifiers and true targets.

    Returns:
    --------
    float
        Mean NDCG@5 score for the ensemble predictions.
    """
    df = base_model_preds[['user_id', 'anchor_order_number', 'label_reordered']].copy()
    df['ensemble_preds'] = ensemble_preds
    
    ndcg_at_k = calculate_ndcg_at_k(df, 'ensemble_preds', 'label_reordered', ['user_id', 'anchor_order_number'], k=5)
    
    return ndcg_at_k
    

def calculate_recall_at_k(df, ensemble_preds, target_col, group_cols, k=5):
    """
    Recall at k calculator for optuna ensemble weight tuning.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with predictions and targets.
    ensemble_preds : str
        Prediction column name.
    target_col : str
        Target label column name.
    group_cols : list of str
        Grouping hierarchy columns.
    k : int, default=5
        Top-k cutoff value.

    Returns:
    --------
    float
        Mean Recall@K across valid orders.
    """
    sorted_df = df.sort_values(by=group_cols + [ensemble_preds], ascending=[True, True, False])
    top_k_recommendations = sorted_df.groupby(group_cols).head(k)
    
    true_positives_per_order = top_k_recommendations.groupby(group_cols)[target_col].sum()
    actual_positives_per_order = sorted_df.groupby(group_cols)[target_col].sum()

    #Only look at orders where reorders > 0 (actual positives > 0)
    valid_orders = actual_positives_per_order > 0
    tp_valid = true_positives_per_order[valid_orders]
    ap_valid = actual_positives_per_order[valid_orders]
    
    recall_per_order = tp_valid / ap_valid
    
    return recall_per_order.mean()

# Make sure these imports are at the very top of your utility_functions.py file:
# import pandas as pd
# import numpy as np
# from sklearn.metrics import ndcg_score

def process_order(order, k):
    """
    Calculates NDCG@K for a single user order group.

    Parameters:
    -----------
    order : pd.DataFrame
        A subset DataFrame corresponding to a single order group containing 'label' and 'ranking_score'.
    k : int
        The cutoff rank for evaluation (e.g., k=5).

    Returns:
    --------
    float or np.nan
        The calculated NDCG@K score for the order, or np.nan if the order has 1 or fewer reorders.
    """
    
    # Count actual positive purchases (reorders) in this specific order
    total_reorders = order['label'].sum()
        
    # If there are no reorders or only 1 reorder, ignore this order.
    # This avoids dividing by zero, returning 0, and unfairly penalizing the model.
    if total_reorders <= 1:
        return np.nan
            
    # Reshape to a 2D array (1, n_items) to satisfy scikit-learn's ndcg_score API
    labels_2d = order['label'].values.reshape(1, -1)
    ranking_scores_2d = order['ranking_score'].values.reshape(1, -1)
        
    # Calculate NDCG@K
    return ndcg_score(labels_2d, ranking_scores_2d, k=k)


def calculate_ndcg_at_k(df, pred_col, target_col, group_cols, k=5):
    """
    Stands alone as an explicit, order-by-order NDCG calculator.
    Uses reshaped 2D numpy arrays and strictly ignores single-item/zero-item orders.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing predictions and targets.
    pred_col : str
        Prediction score column name.
    target_col : str
        Ground-truth target column name.
    group_cols : list of str
        Grouping hierarchical columns.
    k : int, default=5
        Top-k cutoff for evaluation.

    Returns:
    --------
    float
        Clean average NDCG score across all valid orders.
    """
    # Create the structured dataframe for ranking
    ndcg_score_df = pd.DataFrame({
        'user_id': df['user_id'],
        'anchor_order_number': df['anchor_order_number'],
        'label': df[target_col],
        'ranking_score': df[pred_col]
    })
        
    # Apply the process_order function to each individual user order group
    # sort=False to maintain processing speed on large dataset
    results = ndcg_score_df.groupby(group_cols, sort=False).apply(process_order, k, include_groups=False)
    
    # Drop the NaN values (the ignored 0/1 item orders) before taking the mean
    valid_scores = results.dropna()
    
    # Return the clean average NDCG score across all valid orders
    return valid_scores.mean()


def calculate_metrics_at_k(df, pred_col, target_col, group_cols, k=5):
    """
    Calculates NDCG, Recall, Precision, and F1 at K.
    Now leverages the independent calculate_ndcg_at_k function for simplicity.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing predictions and true labels.
    pred_col : str
        Prediction score column name.
    target_col : str
        True target label column name.
    group_cols : list of str
        Grouping hierarchical columns.
    k : int, default=5
        Top-k ranking cutoff threshold.

    Returns:
    --------
    dict
        Dictionary mapping metric names ('NDCG@5', 'Recall@5', 'Precision@5', 'F1@5') to their values.
    """
    # 1. Sort by groups and predictions (highest predicted items at the top)
    sorted_df = df.sort_values(by=group_cols + [pred_col], ascending=[True, True, False])
    top_k_recommendations = sorted_df.groupby(group_cols).head(k)
    
    # 2. Calculate True Positives and Actual Positives per order
    true_positives_per_order = top_k_recommendations.groupby(group_cols)[target_col].sum()
    actual_positives_per_order = sorted_df.groupby(group_cols)[target_col].sum()
    
    # 3. Isolate valid orders (where actual positives > 0) to prevent division by zero
    valid_orders = actual_positives_per_order > 0
    tp_valid = true_positives_per_order[valid_orders]
    ap_valid = actual_positives_per_order[valid_orders]
    
    # 4. Calculate Precision, Recall, and F1 at 5
    precision = (tp_valid / k).mean()
    recall = (tp_valid / ap_valid).mean()
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    # 5. Calculated ndcg at 5
    ndcg = calculate_ndcg_at_k(sorted_df, pred_col, target_col, group_cols, k=k)
    
    return {'NDCG@5': ndcg, 'Recall@5': recall, 'Precision@5': precision, 'F1@5': f1}


def pickle_and_stream_to_gcs(model_object, bucket_name, filename):
    """
    Serializes a Python object in-memory and streams it directly to GCS 
    using JIT authentication.
    
    Parameters:
    -----------
    model_object : any
        The trained model or Python object you want to serialize.
    bucket_name : str
        The raw name of your target GCS bucket (e.g., 'my-bucket-name').
    filename : str
        The destination name inside the bucket (e.g., 'lgbm_ranker_base_model.pkl').

    Returns:
    --------
    bool
        True if upload succeeds, False otherwise.
    """
    
    # Local imports
    import io
    import pickle
    import google.auth
    import google.auth.transport.requests
    from google.cloud import storage
    
    try:
        # 1. Force refresh of the VM's metadata token
        credentials, project = google.auth.default()
        auth_request = google.auth.transport.requests.Request()
        credentials.refresh(auth_request)

        # 2. Serialize the object into a standard memory stream buffer (Zero Disk Footprint)
        memory_buffer = io.BytesIO()
        pickle.dump(model_object, memory_buffer)
        memory_buffer.seek(0)  # Reset stream pointer back to the start
        
        # 3. Instantiate the official GCS Client using your fresh token
        client = storage.Client(credentials=credentials, project=project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        
        # 4. Upload with network timeout to prevent infinite hangs
        blob.chunk_size = 5 * 1024 * 1024  # 5MB Chunks
        blob.upload_from_file(
            memory_buffer, 
            content_type='application/octet-stream',
            timeout=(15.0, 90.0)
        )
        
        # Free memory buffer allocation
        memory_buffer.close()

        print(f"🎉 Success! {filename} successfully streamed to Cloud Storage.")
        
        return True

    except Exception as e:
        print(f"❌ Storage execution failed for {filename}. Error details: {e}")
        
        return False

def get_baseline_evaluation_metrics(df, dataset_split= 'validation'):
    """
    Calculate evaluation metrics for baseline heuristic model
    for model_evaluation and financial impact notebooks.

    Parameters:
    -----------
    df : pd.DataFrame
        test_df or val_df subset.
    dataset_split : str, default='validation'
        Label string denoting data split.

    Returns:
    --------
    tuple
        Returns (heuristic_evaluation_metrics, heuristic_recall, heuristic_ndcg).
    """
    heuristic_df = df.copy()

    #fill null values with 9999 to ensure they are not in top 5 products for widget
    heuristic_df['heuristic_score'] = heuristic_df['user_product_cart_position_last_order'].fillna(9999)
    
    # The utility function (calculate_metrics_at_k) sorts predictions descending.
    # Since a lower add_to_cart_order is better (1st is best), multiply by -1 to flip the scale
    heuristic_df['heuristic_score'] = -1 * heuristic_df['heuristic_score']
    
    # Calculate metrics for the heuristic model
    heuristic_metrics = calculate_metrics_at_k(df=heuristic_df, pred_col= 'heuristic_score', 
                                               target_col= 'label_reordered', group_cols= ['user_id', 'anchor_order_number'], k=5)
    
    heuristic_recall = heuristic_metrics['Recall@5']
    heuristic_ndcg = heuristic_metrics['NDCG@5']
    
    print(f"Heuristic Recall@5: {heuristic_recall:.4f}")
    print(f"Heuristic NDCG@5:   {heuristic_ndcg:.4f}")

    # Build heuristic evaluation metrics to append to model evaluation results
    heuristic_evaluation_metrics = pd.DataFrame([{
        'Split': dataset_split,
        'Type' : 'Heuristic Baseline',
        'Model': 'Heuristic Baseline (Prior Cart Position)',
        'Precision@5': heuristic_metrics.get('Precision@5'),
        'Recall@5': heuristic_recall,
        'F1@5': heuristic_metrics.get('F1@5'),
        'NDCG@5': heuristic_ndcg  # Storing NDCG in the ranking map slot
    }])
    
    return heuristic_evaluation_metrics, heuristic_recall, heuristic_ndcg