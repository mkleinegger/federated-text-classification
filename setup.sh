#!/bin/bash
set -e

ENV_NAME="federated-learning"

python3 -m venv $ENV_NAME

source $ENV_NAME/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"

echo "Setup complete. To activate the virtual environment, run 'source $ENV_NAME/bin/activate'"
