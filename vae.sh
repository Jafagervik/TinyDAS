#!/bin/bash

#SBATCH -p GPUQ
#SBATCH -J vae
#SBATCH --account=ie-idi
#SBATCH -t 06:00:00
#SBATCH -N 1 # One node
#SBATCH --mem=160G
#SBATCH --constraint="gpu80g"
#SBATCH --gres=gpu:2
#SBATCH --output=logs/vae.txt
#SBATCH --error=logs/vae.err


module purge
module load Python/3.11.5-GCCcore-13.2.0
source /cluster/home/jorgenaf/pm/bin/activate

export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
python /cluster/home/jorgenaf/TinyDAS/run.py --model VAE --mode train 

