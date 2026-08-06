#!/bin/bash

# Exit immediately if any command fails
set -e

# Dynamically pull the root_loc variable from your config file
ROOT_LOC=$(python3 -c "from config.config import root_loc; print(root_loc)")
API_FILE="$ROOT_LOC/config/enabled_apis.txt"

if [ ! -f "$API_FILE" ]; then
    echo "❌ Error: API specification file not found at $API_FILE"
    exit 1
fi

echo "🚀 Enabling GCP APIs specified in $API_FILE..."

# Extract non-empty, non-commented API names and pass them to gcloud
APIS=$(grep -v '^#' "$API_FILE" | grep -v '^\s*$' | tr '\n' ' ')

gcloud services enable $APIS

echo "✅ All specified GCP APIs successfully enabled!"