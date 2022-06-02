#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=1200
#SBATCH --time 24:00:00
#SBATCH --gres gpu:v100:1

module load opence/1.5.1
cd ~/elasticc/ && conda activate elasticc

parsnip_train model/parsnip-elasticc-extragal-transients.pt data/{CART,ILOT,KN_B19,KN_K17,PISN,SLSN-I+host,SNIa-91bg,SNIa-SALT2,SNIax,SNIb+HostXT_V19,SNIb-Templates,SNIcBL+HostXT_V19,SNIc+HostXT_V19,SNIc-Templates,SNIIb+HostXT_V19,SNII+HostXT_V19,SNIIn+HostXT_V19,SNII-NMF,SNIIn-MOSFIT,SNII-Templates,TDE}_count1000.hdf5 --threads=4 --device=cuda
