#!/bin/bash

set -e

EXPS=(
    # VGG CELLS
    vgg-cells/segmentation/unet_{16,32,50}_run-{0..9}
    vgg-cells/detection/fasterrcnn_{16,32,50}_run-{0..9}
    vgg-cells/density/csrnet_{16,32,50}_run-{0..9}
    # MBM CELLS
    mbm-cells/segmentation/unet_{5,10,15}_run-{0..9}
    mbm-cells/detection/fasterrcnn_{5,10,15}_run-{0..9}
    mbm-cells/density/csrnet_{5,10,15}_run-{0..9}
    # Nuclei
    nuclei-cells/segmentation/unet_50_fold-{1..2}-of-2
    nuclei-cells/detection/fasterrcnn_50_fold-{1..2}-of-2
    nuclei-cells/density/csrnet_50_fold-{1..2}-of-2
)

# Train & Evaluate
for EXP in ${EXPS[@]}; do
    python train.py experiment=$EXP
    python evaluate.py runs/experiment=$EXP
done
