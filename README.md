# Parsnip model traing for Elasticc

### Create venv

You know how

### Install deps
```sh
python3 -mpip install -r requirements.txt
```

### Create `lcdata` HDF5 files

```sh
for DIR in `ls FULL_ELASTICC_TRAIN | sort`; do
    MODEL=${DIR#ELASTICC_TRAIN_}
    python3 ./create_dataset.py -t ${MODEL} -i FULL_ELASTICC_TRAIN/${DIR} -o data/${MODEL}.hdf5
    python3 ./create_dataset.py -t ${MODEL} -i FULL_ELASTICC_TRAIN/${DIR} -o data/${MODEL}_z0.001.hdf5 --fixed-z=0.001
done

for DIR in `ls FULL_ELASTICC_TRAIN | sort`; do
    MODEL=${DIR#ELASTICC_TRAIN_}
    python3 ./create_dataset.py -t ${MODEL} -i FULL_ELASTICC_TRAIN/${DIR} -o data/${MODEL}_count1000.hdf5 --count=1000
    python3 ./create_dataset.py -t ${MODEL} -i FULL_ELASTICC_TRAIN/${DIR} -o data/${MODEL}_count1000_z0.001.hdf5 --count=1000 --fixed-z=0.001
done
```

### Train Parsnip model for transients only
```sh
parsnip_train model/parsnip-elasticc-extragal-SNe.pt data/{SNIa-91bg,SNIa-SALT2,SNIax,SNIb+HostXT_V19,SNIb-Templates,SNIcBL+HostXT_V19,SNIc+HostXT_V19,SNIc-Templates,SNIIb+HostXT_V19,SNII+HostXT_V19,SNIIn+HostXT_V19,SNII-NMF,SNIIn-MOSFIT,SNII-Templates}.hdf5 --threads=4 --device=cuda --batch_size=128 --learning_rate=1e-5 --min_learning_rate=1e-8

parsnip_train model/parsnip-elasticc-extragal-transients.pt data/{CART,ILOT,KN_B19,KN_K17,PISN,SLSN-I+host,SNIa-91bg,SNIa-SALT2,SNIax,SNIb+HostXT_V19,SNIb-Templates,SNIcBL+HostXT_V19,SNIc+HostXT_V19,SNIc-Templates,SNIIb+HostXT_V19,SNII+HostXT_V19,SNIIn+HostXT_V19,SNII-NMF,SNIIn-MOSFIT,SNII-Templates}.hdf5 --threads=4 --device=cuda --batch_size=128 --learning_rate=1e-5 --min_learning_rate=1e-8

parsnip_train model/parsnip-elasticc-all-transients.pt data/{CART,ILOT,KN_B19,KN_K17,PISN,SLSN-I+host,SNIa-91bg,SNIa-SALT2,SNIax,SNIb+HostXT_V19,SNIb-Templates,SNIcBL+HostXT_V19,SNIc+HostXT_V19,SNIc-Templates,SNIIb+HostXT_V19,SNII+HostXT_V19,SNIIn+HostXT_V19,SNII-NMF,SNIIn-MOSFIT,SNII-Templates,TDE,dwarf_nova_z0.001,Mdwarf-flare_z0.001,uLens-Binary_z0.001,uLens-Single-GenLens_z0.001,uLens-Single_PyLIMA_z0.001}.hdf5 --threads=4 --device=cuda --batch_size=128 --learning_rate=1e-5 --min_learning_rate=1e-8
```