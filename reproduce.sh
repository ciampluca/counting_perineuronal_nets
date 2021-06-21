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
    perineuronal-nets/detection/fasterrcnn_800
    perineuronal-nets/density/csrnet_256
    perineuronal-nets/density/csrnet_320
    perineuronal-nets/density/csrnet_480
    perineuronal-nets/density/csrnet_640
    perineuronal-nets/density/csrnet_800
)

for EXP in ${EXPS[@]}; do
    python train.py experiment=$EXP
done

# VGG CELLS
EXPS=(
    vgg-cells/segmentation/unet
    vgg-cells/detection/fasterrcnn_nms_0.4
    vgg-cells/density/csrnet
)

for EXP in ${EXPS[@]}; do
    python train.py experiment=$EXP
done