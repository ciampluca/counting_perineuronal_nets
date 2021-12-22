#!/bin/bash

VGG_DIR="vgg-cells"
MBM_DIR="mbm-cells"
NUCLEI_DIR="nuclei-cells"

VGG_URL="http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip"
MBM_URL="https://github.com/ieee8023/countception/raw/master/MBM_data.zip"
NUCLEI_URL="https://warwick.ac.uk/fac/cross_fac/tia/data/crchistolabelednucleihe/crchistophenotypes_2016_04_28.zip"

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

# NUCLEI DATASET
if [[ ! -d "${NUCLEI_DIR}" ]]; then
    echo "Downloading and extracting: Nuclei"
    wget --no-clobber "${NUCLEI_URL}" -O nuclei-cells.zip
    unzip nuclei-cells.zip -d "${NUCLEI_DIR}"  && rm nuclei-cells.zip

    echo "Preparing: Nuclei"
    python prepare_data.py --data-name 'nuclei' --data-path "${NUCLEI_DIR}"
    rm -rf "${NUCLEI_DIR}"/CRCHistoPhenotypes_2016_04_28 && rm -rf "${NUCLEI_DIR}"/__MACOSX
fi

echo "DONE"