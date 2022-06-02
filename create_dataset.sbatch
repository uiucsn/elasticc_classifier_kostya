#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=1200
#SBATCH --time 1:30:00
#SBATCH --array=0-31

DIR_NAMES=($(ls FULL_ELASTICC_TRAIN | sort))

export DIR_NAME=${DIR_NAMES[$SLURM_ARRAY_TASK_ID]}
export MODEL=${DIR_NAME#ELASTICC_TRAIN_}
export DIR="FULL_ELASTICC_TRAIN/ELASTICC_TRAIN_${MODEL}"

module load opence/1.5.1
cd ~/elasticc/ && conda activate elasticc

python3 ./create_dataset.py -t ${MODEL} -i ${DIR} -o data/${MODEL}_count1000.hdf5 --count=1000
python3 ./create_dataset.py -t ${MODEL} -i ${DIR} -o data/${MODEL}_count1000_z0.001.hdf5 --count=1000 --fixed-z=0.001
