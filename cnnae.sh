#!/bin/bash

#SBATCH -p GPUQ
#SBATCH -J tinydas_cnnae
#SBATCH --account=ie-idi
#SBATCH -t 02:00:00
#SBATCH -N 1 # One node
#SBATCH --mem=160G
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu40g|gpu80g"
#SBATCH --output=logs/cnnae.txt # Log file

module purge
module load Python/3.11.5-GCCcore-13.2.0
source /cluster/home/jorgenaf/master/bin/activate

# WORKDIR = ${SLURM_SUBMIT_DIR}
# cd ${WORKDIR}
python main.py -t train -m cnnae -g 4 -d 