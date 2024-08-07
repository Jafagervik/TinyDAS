BATCH -p GPUQ
#SBATCH -J tinydas_cae
#SBATCH --account=ie-idi
#SBATCH -t 01:00:00
#SBATCH -N 1 # One node
#SBATCH --mem=80G
#SBATCH --constraint="gpu80g"
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cae.txt
#SBATCH --error=logs/cae.err

module purge
module load Python/3.11.5-GCCcore-13.2.0
source /cluster/home/jorgenaf/master/bin/activate

export PYTHONUNBUFFERED=1
python main.py -t detect -m ae -g 1

