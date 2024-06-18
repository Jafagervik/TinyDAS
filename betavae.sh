#!/bin/bash

#SBATCH -p GPUQ
#SBATCH -J tinydas_betavae
#SBATCH --account=ie-idi
#SBATCH -t 01:00:00
#SBATCH -N 1 # One node
#SBATCH --mem=160G
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu40g|gpu80g"
#SBATCH --output=logs/cnnae.txt 
#SBATCH --error=logs/cnnae.err 

module purge
module load Python/3.11.5-GCCcore-13.2.0
source /cluster/home/jorgenaf/master/bin/activate

export PYTHONUNBUFFERED=1
srun python main.py -t train -m betavae -g 4 -d 