#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=80
#SBATCH --mem-per-cpu=1200
#SBATCH --time 4:00:00
#SBATCH --array=0-31

DIR_NAMES=($(ls FULL_ELASTICC_TRAIN | sort))

export DIR_NAME=${DIR_NAMES[$SLURM_ARRAY_TASK_ID]}
export MODEL=${DIR_NAME#ELASTICC_TRAIN_}
export DIR="FULL_ELASTICC_TRAIN/ELASTICC_TRAIN_${MODEL}"

module load opence/1.6.1
cd ~/elasticc/ && conda activate elasticc-opence-v1.6.1

python3 ./create_dataset.py --parallel -t ${MODEL} -i ${DIR} --split-prob=0.2 -o data_split/${MODEL}.hdf5
# python3 ./create_dataset.py -t ${MODEL} -i ${DIR} --split-prob=0.2 -o data_split/${MODEL}_z0.001.hdf5 --fixed-z=0.001
# python3 ./create_dataset.py --parallel -t ${MODEL} -i ${DIR} --split-prob=0.2 -o data_split/${MODEL}_count5000.hdf5 --count=5000
# python3 ./create_dataset.py -t ${MODEL} -i ${DIR} --split-prob=0.2 -o data_split/${MODEL}_count5000_z0.001.hdf5 --count=5000 --fixed-z=0.001
