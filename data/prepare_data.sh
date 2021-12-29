#!/bin/bash

VGG_DIR="vgg-cells"
MBM_DIR="mbm-cells"
BCD_DIR="bcd-cells"
ADIPOCYTE_DIR="adipocyte-cells"
PNN_DIR="perineuronal-nets"

VGG_URL="http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip"
MBM_URL="https://github.com/ieee8023/countception/raw/master/MBM_data.zip"
BCD_URL="http://datino.isti.cnr.it/BCData.zip"
ADIPOCYTE_URL="http://datino.isti.cnr.it/Adipocyte.zip"
PNN_URL="https://zenodo.org/record/5567032/files/pnn.zip?download=1"


# VGG DATASET
if [[ ! -d "${VGG_DIR}" ]]; then
    echo "Downloading and extracting: VGG"
    wget --no-clobber "${VGG_URL}" -O vgg-cells.zip
    unzip vgg-cells.zip -d "${VGG_DIR}"  && rm vgg-cells.zip

    echo "Preparing: VGG"
    python prepare_data.py --data-name 'VGG' --data-path "${VGG_DIR}"
    mkdir -p "${VGG_DIR}/imgs" && mv "${VGG_DIR}"/*cell.png "${VGG_DIR}"/imgs && rm "${VGG_DIR}"/*.png
fi

# MBM DATASET
if [[ ! -d "${MBM_DIR}" ]]; then
    echo "Downloading and extracting: MBM"
    wget --no-clobber "${MBM_URL}" -O mbm-cells.zip
    unzip -j mbm-cells.zip 'MBM_data/*' -d "${MBM_DIR}" && rm mbm-cells.zip

    echo "Preparing: MBM"
    for IMGPATH in ${MBM_DIR}/*.png; do
        if [[ "$IMGPATH" =~ "dots" ]]; then
            # move '_dots' at the end
            NEW_PATH=${IMGPATH//_dots/}
            NEW_PATH="${NEW_PATH%.png}_dots.png"
        else
            # add '_cell' at the end
            NEW_PATH="${IMGPATH%.png}_cell.png"
        fi
        echo "${IMGPATH} --> ${NEW_PATH}"
        mv "${IMGPATH}" "${NEW_PATH}"
    done
    python prepare_data.py --data-name 'MBM' --data-path "${MBM_DIR}"
    mkdir -p "${MBM_DIR}/imgs" && mv "${MBM_DIR}"/*cell.png "${MBM_DIR}"/imgs && rm "${MBM_DIR}"/*.png
fi

# BCD DATASET
if [[ ! -d "${BCD_DIR}" ]]; then
    echo "Downloading and extracting: BCD"
    wget --no-clobber "${BCD_URL}" -O bcd-cells.zip
    unzip bcd-cells.zip -d "${BCD_DIR}"  && rm bcd-cells.zip

    echo "Preparing: BCD"
    python prepare_data.py --data-name 'BCD' --data-path "${BCD_DIR}/BCData/images/train"
    python prepare_data.py --data-name 'BCD' --data-path "${BCD_DIR}/BCData/images/validation"
    python prepare_data.py --data-name 'BCD' --data-path "${BCD_DIR}/BCData/images/test"
    rm -rf "${BCD_DIR}"/BCData
fi

# ADIPOCYTE DATASET
if [[ ! -d "${ADIPOCYTE_DIR}" ]]; then
    echo "Downloading and extracting: ADIPOCYTE"
    wget --no-clobber "${ADIPOCYTE_URL}" -O adipocyte-cells.zip
    unzip adipocyte-cells.zip -d "${ADIPOCYTE_DIR}"  && rm adipocyte-cells.zip

    echo "Preparing: ADIPOCYTE"
    python prepare_data.py --data-name 'Adipocyte' --data-path "${ADIPOCYTE_DIR}"
    rm -rf "${ADIPOCYTE_DIR}"/Adipocyte_cells
fi

# DOWNLOAD PNN DATASET
if [[ ! -d "${PNN_DIR}" ]]; then
    echo "Downloading and extracting: PNN"
    wget --no-clobber "${PNN_URL}" -O pnn.zip
    unzip pnn.zip -d "${PNN_DIR}" && rm pnn.zip
    # converts TIFFs to HDF5 for the PNN dataset
    python convert_tifs_to_hdf5.py
fi


echo "DONE"