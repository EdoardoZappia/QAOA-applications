#!/bin/bash
#SBATCH --job-name=QAOA_maxcut
#SBATCH --output=QAOA_maxcut_%j.log
#SBATCH --error=QAOA_maxcut_%j.err
#SBATCH --time=02:00:00          
#SBATCH --partition=THIN         
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=4        
#SBATCH --gres=gpu:1             
#SBATCH --mem=16G                

# Carica i moduli necessari (aggiusta in base alla tua configurazione)
module load python/3.9
module load tensorflow
module load keras
module load pennylane

# Attiva l'ambiente virtuale
source /path/to/your/venv/bin/activate

# Esegui lo script Python
python /path/to/your/script.py

# Disattiva l'ambiente virtuale
deactivate
