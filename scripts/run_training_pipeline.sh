#!/bin/bash

# Exit immediately if any command in the chain fails
set -e 

#Pull root directory from config.py
ROOT_LOC=$(python3 -c "from config.config import root_loc; print(root_loc)")

#Export root_loc to PYTHONPATH so Python finds 'config' and 'src'
export PYTHONPATH="$ROOT_LOC:$PYTHONPATH"

# Define the exact Python interpreter used by Jupyter kernel for Notebooks
PY_ENV=$(python3 -c "from config.config import PY_ENV; print(PY_ENV)")

echo "🚀 Starting ML Training & Tuning Pipeline using active Jupyter environment..."
echo "📍 Project Root set to: $ROOT_LOC"

# Execute the base models sequentially using the explicit virtual environment
echo "🚀 Starting LGBM Ranker Training & Tuning..."

$PY_ENV src/base_models/train_lgbm_ranker.py

echo "✅ LGBM Ranker Training & Tuning Complete..."
echo "🚀 Starting LGBM Classifier Training & Tuning..."

$PY_ENV src/base_models/train_lgbm_classifier.py

echo "✅ LGBM Classifier Training & Tuning Complete..."
echo "🚀 Starting XGBOOST Ranker Training & Tuning..."

$PY_ENV src/base_models/train_xgboost_ranker.py

echo "✅ XGBOOST Ranker Training & Tuning Complete..."
echo "🚀 Starting XGBOOST Classifier Training & Tuning..."

$PY_ENV src/base_models/train_xgboost_classifier.py

echo "✅ XGBOOST Classifier Training & Tuning Complete..."
echo "🚀 Starting Ensemble Weight Tuning..."

# Execute the ensembling script
$PY_ENV src/ensemble/ensemble.py

echo "✅ Ensemble Weight Tuning Complete..."

echo "✅ Pipeline execution complete!"
