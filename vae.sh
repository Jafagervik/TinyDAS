#!/bin/bash

#SBATCH -p GPUQ
#SBATCH -J tinydas_vae
#SBATCH --account=ie-idi
#SBATCH -t 03:00:00
#SBATCH -N 1 # One node
#SBATCH --mem=320G
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu80g"
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jorgenaf@stud.ntnu.no
#SBATCH --output=logs/vae.txt
#SBATCH --error=logs/vae.err

module purge
module load Python/3.11.5-GCCcore-13.2.0
source /cluster/home/jorgenaf/master/bin/activate

export PYTHONUNBUFFERED=1
srun python main.py -t train -m vae -g 4