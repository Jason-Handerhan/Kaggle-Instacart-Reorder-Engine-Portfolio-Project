#!/bin/bash

# Exit immediately if any command fails
set -e

# Dynamically pull the PY_ENV variable from your config file
PY_ENV=$(python3 -c "from config.config import PY_ENV; print(PY_ENV)")

echo "🚀 Installing requirements into environment: $PY_ENV"

# Install requirements using the target Python executable's pip
$PY_ENV -m pip install -r requirements.txt

echo "✅ All dependencies successfully installed into $PY_ENV!"