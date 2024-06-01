#!/bin/bash

#SBATCH --job-name=QAOA_maxcut
#SBATCH --output=QAOA_maxcut_%j.log
#SBATCH --error=QAOA_maxcut_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Scarica e installa Miniconda
echo "Downloading Miniconda installer..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh

# Verifica che il file sia stato scaricato correttamente
echo "Verifying the integrity of the installer..."
EXPECTED_CHECKSUM="2bf16dc434370d374e9eb22b5b2f6555"
DOWNLOADED_CHECKSUM=$(md5sum Miniconda3-latest-Linux-x86_64.sh | awk '{ print $1 }')

if [ "$EXPECTED_CHECKSUM" != "$DOWNLOADED_CHECKSUM" ]; then
    echo "Error: Checksum mismatch for the Miniconda installer."
    echo "Expected: $EXPECTED_CHECKSUM"
    echo "Got: $DOWNLOADED_CHECKSUM"
    exit 1
fi

# Installa Miniconda
echo "Installing Miniconda..."
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Inizializza conda
echo "Initializing conda..."
source $HOME/miniconda3/etc/profile.d/conda.sh
conda config --set auto_activate_base false

# Crea e attiva l'ambiente conda
echo "Creating and activating conda environment..."
conda create --yes --name deep_learning python=3.9
conda activate deep_learning

# Installa i pacchetti necessari
echo "Installing necessary packages..."
conda install --yes tensorflow keras numpy pennylane

# Naviga alla directory dello script
cd /orfeo/cephfs/home/dssc/ezappia/QAOA-applications/QAOA.py

# Esegui lo script Python
echo "Running the Python script..."
python QAOA.py

# Disattiva l'ambiente conda
echo "Deactivating conda environment..."
conda deactivate
