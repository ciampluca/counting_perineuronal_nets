# Data Download and Preparation

```bash
# be sure to be in this directory
cd data/

# downloads and extracts VGG, MBM, and PNN datasets
./download_data.sh

# converts TIFFs to HDF5 for the PNN dataset
python convert_tifs_to_hdf5.py
```