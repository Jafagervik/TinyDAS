#!/bin/bash

#SBATCH -p GPUQ
#SBATCH -J tinydas_vae
#SBATCH --account=ie-idi
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --output=logs/vae.txt
#SBATCH --error=logs/vae.err

module purge
module load Python/3.11.5-GCCcore-13.2.0
source /cluster/home/jorgenaf/master/bin/activate

export PYTHONUNBUFFERED=1
python main.py -t train -m vae -g 1