#!/bin/bash

VGG_DIR="vgg-cells"
MBM_DIR="mbm-cells"
PNN_DIR="perineuronal-nets"

VGG_URL="http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip"
MBM_URL="https://github.com/ieee8023/countception/raw/master/MBM_data.zip"
PNN_URL="https://zenodo.org/record/5567032/files/pnn.zip?download=1"

# DOWNLOAD VGG CELLS DATASET
if [[ ! -d "${VGG_DIR}" ]]; then
    echo "Downloading and extracting: VGG"
    wget --no-clobber "${VGG_URL}" -O vgg-cells.zip
    unzip vgg-cells.zip -d "${VGG_DIR}"  && rm vgg-cells.zip
fi

# DOWNLOAD MBM CELLS DATASET
if [[ ! -d "${MBM_DIR}" ]]; then
    echo "Downloading and extracting: MBM"
    wget --no-clobber "${MBM_URL}" -O mbm-cells.zip
    unzip -j mbm-cells.zip 'MBM_data/*' -d "${MBM_DIR}" && rm mbm-cells.zip

    # rename files
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
fi

# DOWNLOAD PNN DATASET
if [[ ! -d "${PNN_DIR}" ]]; then
    echo "Downloading and extracting: PNN"
    wget --no-clobber "${PNN_URL}" -O pnn.zip
    unzip pnn.zip -d "${PNN_DIR}" && rm pnn.zip
fi

echo "DONE"