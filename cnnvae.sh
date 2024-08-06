#!/bin/bash

#SBATCH -p GPUQ
#SBATCH -J cnnvae
#SBATCH --account=ie-idi
#SBATCH -t 08:00:00
#SBATCH -N 1 # One node
#SBATCH --mem=320G
#SBATCH --constraint="gpu80g"
#SBATCH --gres=gpu:4
#SBATCH --output=logs/cnnvae.txt
#SBATCH --error=logs/cnnvae.err



module purge
module load Python/3.11.5-GCCcore-13.2.0
source /cluster/home/jorgenaf/pm/bin/activate

export PYTHONUNBUFFERED=1
python /cluster/home/jorgenaf/TinyDAS/run.py --model CNNVAE --mode train  