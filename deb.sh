#!/bin/bash

#SBATCH -p GPUQ
#SBATCH -J tinydas_debug_vae
#SBATCH --account=ie-idi
#SBATCH -t 01:00:00
#SBATCH -N 1 # One node
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu40g|gpu80g"
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jorgenaf@stud.ntnu.no
#SBATCH --output=logs/debug.txt
#SBATCH --error=logs/debug.err

module purge
module load Python/3.11.5-GCCcore-13.2.0
source /cluster/home/jorgenaf/master/bin/activate

export PYTHONUNBUFFERED=1
srun python main.py -t train -m vae -g 2
