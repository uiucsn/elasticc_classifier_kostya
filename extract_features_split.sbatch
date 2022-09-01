#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=1200
#SBATCH --gres gpu:v100:1
#SBATCH --time 24:00:00
#SBATCH --array=0-31

export RUST_BACKTRACE=1

DIR_NAMES=($(ls FULL_ELASTICC_TRAIN | sort))
DIR_NAME=${DIR_NAMES[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME=${DIR_NAME#ELASTICC_TRAIN_}
export DATA="data_split/${MODEL_NAME}.hdf5"

set -x

module load opence/1.6.1
cd ~/elasticc/ && conda activate elasticc-opence-v1.6.1

python3 ./extract_features.py -i ${DATA} -m model_split/parsnip-elasticc-extragal-SNe.pt -o features_split --device=cuda --s2n=5.0
