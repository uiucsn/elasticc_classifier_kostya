#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=80
#SBATCH --mem-per-cpu=1200
#SBATCH --time 24:00:00
#SBATCH --gres gpu:v100:1

module load opence/1.6.1
cd ~/elasticc/ && conda activate elasticc-opence-v1.6.1

nvidia-smi

python3 ./train_classifier.py --features=features_split/ --figures=figures_split/ --output=model_split/
