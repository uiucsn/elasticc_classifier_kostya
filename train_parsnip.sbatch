#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=1200
#SBATCH --time 24:00:00
#SBATCH --gres gpu:v100:4

module load opence/1.6.1
cd ~/elasticc/ && conda activate elasticc-opence-v1.6.1

watch_nvidia_smi () {
  nvidia-smi
  sleep 60
}

watch_nvidia_smi &

parsnip_train model/parsnip-elasticc-extragal-SNe.pt data/{SNIa-91bg,SNIa-SALT2,SNIax,SNIb+HostXT_V19,SNIb-Templates,SNIcBL+HostXT_V19,SNIc+HostXT_V19,SNIc-Templates,SNIIb+HostXT_V19,SNII+HostXT_V19,SNIIn+HostXT_V19,SNII-NMF,SNIIn-MOSFIT,SNII-Templates}.hdf5 --threads=4 --device=cuda --batch_size=128 --learning_rate=1e-5 --min_learning_rate=1e-8 --zeropoint=27.5 --data_parallelism
