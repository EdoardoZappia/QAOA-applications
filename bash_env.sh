#!/bin/bash

# Install Miniconda if not already installed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda config --set auto_activate_base false

# Create and activate the conda environment
conda create --yes --name deep_learning python=3.9
conda activate deep_learning

# Install necessary packages
conda install --yes tensorflow keras numpy pennylane

# Navigate to the script directory


# Run the Python script
python QAOA.py

# Deactivate the conda environment
conda deactivate
