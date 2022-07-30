#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=1200
#SBATCH --gres gpu:v100:1
#SBATCH --time 06:00:00
#SBATCH --array=0-31

export RUST_BACKTRACE=1

DIR_NAMES=($(ls FULL_ELASTICC_TRAIN | sort))
DIR_NAME=${DIR_NAMES[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME=${DIR_NAME#ELASTICC_TRAIN_}
export DATA="data/${MODEL_NAME}.hdf5"

set -x

module load opence/1.5.1
cd ~/elasticc/ && conda activate elasticc

python3 ./extract_features.py -i ${DATA} -m model/parsnip-elasticc-extragal-SNe.pt -o features --device=cuda --s2n=5.0
