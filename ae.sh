#!/bin/bash

#SBATCH -p GPUQ
#SBATCH -J tinydas_ae
#SBATCH --account=ie-idi
#SBATCH -t 04:00:00
#SBATCH -N 1 # One node
#SBATCH --mem=320G
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu80g"
#SBATCH --output=logs/ae.out        # Output log file (standard output)
#SBATCH --error=logs/ae.err         # Error log file (standard error)

module purge
module load Python/3.11.5-GCCcore-13.2.0
source /cluster/home/jorgenaf/master/bin/activate

python main.py -t train -m ae -g 4