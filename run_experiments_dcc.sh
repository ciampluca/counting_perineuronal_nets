#!/bin/bash

EXPS=(
    dcc-cells/density/csrnet_70-30_run-0
    #dcc-cells/density/csrnet_70-30_run-1
    #dcc-cells/density/csrnet_70-30_run-2
    #dcc-cells/density/csrnet_70-30_run-3
    #dcc-cells/density/csrnet_70-30_run-4
    #dcc-cells/density/csrnet_70-30_run-5
    #dcc-cells/density/csrnet_70-30_run-6
    #dcc-cells/density/csrnet_70-30_run-7
    #dcc-cells/density/csrnet_70-30_run-8
    #dcc-cells/density/csrnet_70-30_run-9
    dcc-cells/detection/fasterrcnn_70-30_run-0
    dcc-cells/detection/fasterrcnn_70-30_run-1
    dcc-cells/detection/fasterrcnn_70-30_run-2
    dcc-cells/detection/fasterrcnn_70-30_run-3
    dcc-cells/detection/fasterrcnn_70-30_run-4
    dcc-cells/detection/fasterrcnn_70-30_run-5
    dcc-cells/detection/fasterrcnn_70-30_run-6
    dcc-cells/detection/fasterrcnn_70-30_run-7
    dcc-cells/detection/fasterrcnn_70-30_run-8
    dcc-cells/detection/fasterrcnn_70-30_run-9
)


for EXP in ${EXPS[@]}; do
    CUDA_VISIBLE_DEVICES=0 python train.py experiment=$EXP
    #CUDA_VISIBLE_DEVICES=0 python evaluate.py --best-on-metric count/game-0 --debug --data-root /mnt/datino/Cells/DCC_cells/test --test-split all ./runs/experiment=$EXP
done

