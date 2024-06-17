#!/bin/bash

#SBATCH -J job                  # Sensible name for the job
#SBATCH --account=ie-idi
#SBATCH -N 1                    # Allocate 2 nodes for the job
#SBATCH -t 00:30:00             # Upper time limit for the job
#SBATCH --mem=64G
#SBATCH -p CPUQ
#SBATCH --output=log.txt 

module purge
module load Julia/1.9.3-linux-x86_64

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
srun julia scripts/split_files.jl