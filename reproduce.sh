#!/bin/bash

# PERINEURAL NETS
EXPS=(
    perineuronal-nets/segmentation/unet_256
    perineuronal-nets/segmentation/unet_320
    perineuronal-nets/segmentation/unet_480
    perineuronal-nets/segmentation/unet_640
    perineuronal-nets/segmentation/unet_800
    perineuronal-nets/detection/fasterrcnn_256
    perineuronal-nets/detection/fasterrcnn_320
    perineuronal-nets/detection/fasterrcnn_480
    perineuronal-nets/detection/fasterrcnn_640
    perineuronal-nets/detection/fasterrcnn_640_nococo
    perineuronal-nets/detection/fasterrcnn_800
    perineuronal-nets/detection/fasterrcnn_800_another_seed
    perineuronal-nets/density/csrnet_256
    perineuronal-nets/density/csrnet_320
    perineuronal-nets/density/csrnet_480
    perineuronal-nets/density/csrnet_640
    perineuronal-nets/density/csrnet_800
)

for EXP in ${EXPS[@]}; do
    python train.py experiment=$EXP
    python evaluate.py runs/experiment=$EXP
done

# VGG CELLS
EXPS=(
    vgg-cells/density/csrnet_8_92
    vgg-cells/density/csrnet_16_84
    vgg-cells/density/csrnet_32_68
    vgg-cells/density/csrnet_50_50
    vgg-cells/density/csrnet_64_36
    vgg-cells/density/csrnet_80_20
)

for EXP in ${EXPS[@]}; do
    python train.py experiment=$EXP
    python evaluate.py runs/experiment=$EXP
done