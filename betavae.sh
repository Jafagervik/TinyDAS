#!/bin/bash

#SBATCH -p GPUQ
#SBATCH -J tinydas_betavae
#SBATCH --account=ie-idi
#SBATCH -t 02:00:00
#SBATCH -N 1 # One node
#SBATCH --mem=320G
#SBATCH --constraint="gpu80g"
#SBATCH --gres=gpu:4
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jorgenaf@stud.ntnu.no
#SBATCH --output=logs/betavae.txt
#SBATCH --error=logs/betavae.err

module purge
module load Python/3.11.5-GCCcore-13.2.0
source /cluster/home/jorgenaf/master/bin/activate

export PYTHONUNBUFFERED=1
python main.py -t train -m betavae -g 4
