#!/bin/bash

VGG_DIR="vgg-cells"
MBM_DIR="mbm-cells"
DCC_DIR="dcc-cells"
NUCLEI_DIR="nuclei-cells"
BCD_DIR="bcd-cells"
HeLa_DIR="hela-cells"
PSU_DIR="psu-cells"
ADIPOCYTE_DIR="adipocyte-cells"

VGG_URL="http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip"
MBM_URL="https://github.com/ieee8023/countception/raw/master/MBM_data.zip"
DCC_URL="http://datino.isti.cnr.it/DCCData.zip"
NUCLEI_URL="https://warwick.ac.uk/fac/cross_fac/tia/data/crchistolabelednucleihe/crchistophenotypes_2016_04_28.zip"
BCD_URL="http://datino.isti.cnr.it/BCData.zip"
HeLa_URL="https://www.robots.ox.ac.uk/~vgg/software/cell_detection/downloads/CellDetect_v1.0.tar.gz"
PSU_URL="https://scholarsphere.psu.edu/resources/232fd268-3ad5-404f-b25f-4d11eb4dd9db/downloads/4757"
ADIPOCYTE_URL="http://datino.isti.cnr.it/Adipocyte.zip"


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

# DCC DATASET
if [[ ! -d "${DCC_DIR}" ]]; then
    echo "Downloading and extracting: DCC"
    wget --no-clobber "${DCC_URL}" -O dcc-cells.zip
    unzip dcc-cells.zip -d "${DCC_DIR}"  && rm dcc-cells.zip

    echo "Preparing: DCC"
    for IMGPATH in ${DCC_DIR}/trainval/images/*.jpg; do
        # add '_cell' at the end
        IMGNAME="$(basename "${IMGPATH}")"
        NEW_IMGNAME="${IMGNAME%.jpg}_cell.jpg"
        NEW_PATH="${DCC_DIR}/trainval/${NEW_IMGNAME}"
        echo "${IMGPATH} --> ${NEW_PATH}"
        mv "${IMGPATH}" "${NEW_PATH}" 
    done
    rm -rf "${DCC_DIR}/trainval/images"
    for IMGPATH in ${DCC_DIR}/test/images/*.jpg; do
        # add '_cell' at the end
        IMGNAME="$(basename "${IMGPATH}")"
        NEW_IMGNAME="${IMGNAME%.jpg}_cell.jpg"
        NEW_PATH="${DCC_DIR}/test/${NEW_IMGNAME}"
        echo "${IMGPATH} --> ${NEW_PATH}"
        mv "${IMGPATH}" "${NEW_PATH}"
    done
    rm -rf "${DCC_DIR}/test/images"
    for IMGPATH in ${DCC_DIR}/trainval/GT/*.jpg; do
        # add '_dots' at the end
        IMGNAME="$(basename "${IMGPATH}")"
        NEW_IMGNAME="${IMGNAME%.jpg}_dots.jpg"
        NEW_PATH="${DCC_DIR}/trainval/${NEW_IMGNAME}"
        echo "${IMGPATH} --> ${NEW_PATH}"
        mv "${IMGPATH}" "${NEW_PATH}" 
    done
    rm -rf "${DCC_DIR}/trainval/GT"
    for IMGPATH in ${DCC_DIR}/test/GT/*.jpg; do
        # add '_dots' at the end
        IMGNAME="$(basename "${IMGPATH}")"
        NEW_IMGNAME="${IMGNAME%.jpg}_dots.jpg"
        NEW_PATH="${DCC_DIR}/test/${NEW_IMGNAME}"
        echo "${IMGPATH} --> ${NEW_PATH}"
        mv "${IMGPATH}" "${NEW_PATH}" 
    done
    rm -rf "${DCC_DIR}/test/GT"
    python prepare_data.py --data-name 'DCC' --data-path "${DCC_DIR}/trainval"
    mkdir -p "${DCC_DIR}/trainval/imgs" && mv "${DCC_DIR}/trainval"/*cell.jpg "${DCC_DIR}/trainval/imgs" && rm "${DCC_DIR}/trainval"/*.jpg
    python prepare_data.py --data-name 'DCC' --data-path "${DCC_DIR}/test"
    mkdir -p "${DCC_DIR}/test/imgs" && mv "${DCC_DIR}/test"/*cell.jpg "${DCC_DIR}/test/imgs" && rm "${DCC_DIR}/test"/*.jpg
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

# HeLa DATASET
if [[ ! -d "${HeLa_DIR}" ]]; then
    echo "Downloading and extracting: HeLa"
    wget --no-clobber "${HeLa_URL}" -O hela-cells.tar.gz
    mkdir -p "${HeLa_DIR}"
    tar -xvzf hela-cells.tar.gz --directory "${HeLa_DIR}"  && rm hela-cells.tar.gz

    echo "Preparing: HeLa"
    python prepare_data.py --data-name 'HeLa' --data-path "${HeLa_DIR}/CellDetect_v1.0/phasecontrast/trainPhasecontrast"
    python prepare_data.py --data-name 'HeLa' --data-path "${HeLa_DIR}/CellDetect_v1.0/phasecontrast/testPhasecontrast"
    rm -rf "${HeLa_DIR}"/CellDetect_v1.0
fi

# PSU DATASET
if [[ ! -d "${PSU_DIR}" ]]; then
    echo "Downloading and extracting: PSU"
    wget --no-clobber "${PSU_URL}" -O psu-cells.zip
    unzip psu-cells.zip -d "${PSU_DIR}"  && rm psu-cells.zip

    echo "Preparing: PSU"
    python prepare_data.py --data-name 'PSU' --data-path "${PSU_DIR}"
    rm -rf "${PSU_DIR}"/PSU_dataset
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

echo "DONE"