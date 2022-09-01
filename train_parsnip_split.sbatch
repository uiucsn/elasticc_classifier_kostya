#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=1200
#SBATCH --time 24:00:00
#SBATCH --gres gpu:v100:1

module load opence/1.6.1
cd ~/elasticc/ && conda activate elasticc-opence-v1.6.1

watch_nvidia_smi () {
  while :; do
    nvidia-smi
    top -b -n1 -u$USER
    sleep 60
  done
}

watch_nvidia_smi &
WATCH_PID=$!


parsnip_train model_split/parsnip-elasticc-extragal-SNe.pt data_split/{SNIa-91bg,SNIa-SALT2,SNIax,SNIb+HostXT_V19,SNIb-Templates,SNIcBL+HostXT_V19,SNIc+HostXT_V19,SNIc-Templates,SNIIb+HostXT_V19,SNII+HostXT_V19,SNIIn+HostXT_V19,SNII-NMF,SNIIn-MOSFIT,SNII-Templates}.hdf5 --threads=6 --device=cuda --batch_size=256 --learning_rate=1e-5 --min_learning_rate=1e-9 --zeropoint=27.5

echo "Killing cpu and gpu monitor process"
kill $WATCH_PID
