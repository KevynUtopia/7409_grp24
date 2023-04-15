#!/bin/bash
#SBATCH --gres=gpu:1 --cpus-per-task=4 --mail-type=ALL

## sbatch run.sh
## scancel pid

gpu-interactive
. $HOME/anaconda3/etc/profile.d/conda.sh
# sbatch run.sh
conda activate rl

python main_func.py

exit