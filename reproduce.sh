#!/bin/bash

set -e

# ----------------------------------
# STAGE 1 EXPERIMENTS (LOCALIZATION)
# ----------------------------------

EXPS=(
    # VGG CELLS
    vgg-cells/segmentation/unet_{16,32,50}_run-{0..14}
    vgg-cells/detection/fasterrcnn/fasterrcnn_{16,32,50}_run-{0..14}
    vgg-cells/detection/maskrcnn/maskrcnn_{16,32,50}_run-{0..9}
    vgg-cells/density/csrnet/csrnet_{16,32,50}_run-{0..14}
    vgg-cells/density/fcrn-a/fcrn-a_{16,32,50}_run-{0..9}
    # MBM CELLS
    mbm-cells/segmentation/unet_{5,10,15}_run-{0..14}
    mbm-cells/detection/fasterrcnn/fasterrcnn_{5,10,15}_run-{0..14}
    mbm-cells/detection/maskrcnn/maskrcnn_{5,10,15}_run-{0..9}
    mbm-cells/density/csrnet/csrnet_{5,10,15}_run-{0..14}
    mbm-cells/density/fcrn-a/fcrn-a_{5,10,15}_run-{0..9}
    # ADI CELLS
    adi-cells/segmentation/unet_{10,25,50}_run-{0..14}
    adi-cells/detection/fasterrcnn/fasterrcnn_{10,25,50}_run-{0..14}
    adi-cells/detection/maskrcnn/maskrcnn_{10,25,50}_run-{0..9}
    adi-cells/density/csrnet/csrnet_{10,25,50}_run-{0..14}
    adi-cells/density/fcrn-a/fcrn-a_{10,25,50}_run-{0..9}
    # BCD CELLS
    bcd-cells/segmentation/unet/unet_radius-16_run-{0..9}
    bcd-cells/detection/fasterrcnn/fasterrcnn_side-32_nms-0.6_run-{0..9}
    bcd-cells/detection/maskrcnn/maskrcnn_side-32_nms-0.6_run-{0..9}
    bcd-cells/density/csrnet/csrnet_sigma-16_run-{0..9}
    bcd-cells/density/fcrn-a/fcrn-a_run-{0..9}
    # PNN
    perineuronal-nets/segmentation/unet_{256,320,480,640,800}
    perineuronal-nets/detection/fasterrcnn_{256,320,480,640,800}
    perineuronal-nets/density/csrnet_{256,320,480,640,800}
)

# Train & Evaluate
for EXP in ${EXPS[@]}; do
    python train.py experiment=$EXP
    if  [[ $EXP == bcd* ]]
    then
        if  [[ $EXP == *fcrn-a* ]]
        then
            python evaluate.py runs/experiment=$EXP --debug --test-split all --data-root data/bcd-cells/test --best-on-metric count/game-0/macro
        else
            python evaluate.py runs/experiment=$EXP --debug --test-split all --data-root data/bcd-cells/test
        fi
    else
        if  [[ $EXP == *fcrn-a* ]]
        then
            python evaluate.py runs/experiment=$EXP --debug --best-on-metric count/game-0/macro
        else
            python evaluate.py runs/experiment=$EXP --debug 
        fi
    fi
done

# -----------------------------
# STAGE 2 EXPERIMENTS (SCORING)
# -----------------------------

METHODS=(
    simple_regression
    simple_classification
    ordinal_regression
    pairwise_balanced
)

EXPS_TO_RESCORE=(
    perineuronal-nets/segmentation/unet_320
    perineuronal-nets/detection/fasterrcnn_640
    perineuronal-nets/density/csrnet_640
)

for SEED in 45 62 72 84 95; do
for METHOD in ${METHODS[@]}; do
    # Train & Evaluate
    python train_score.py method=$METHOD seed=$SEED
    python evaluate_score.py runs_score/method=$METHOD,seed=$SEED

    # Rescore Locations Found by Stage 1 Methods
    for EXP in ${EXPS_TO_RESCORE[@]}; do
        OUTPUT="runs/experiment=${EXP}/test_predictions/all_gt_preds_rescored_${METHOD}-seed${SEED}.csv.gz"

        if [ -f "$OUTPUT" ]; then
           echo "SKIPPING EXISTING: $EXP $METHOD $SEED"
           continue
        fi
        
        python score.py \
            runs_score/method=$METHOD,seed=$SEED \
            runs/experiment=${EXP}/test_predictions/all_gt_preds.csv.gz \
            -o ${OUTPUT} \
            -r data/perineuronal-nets/test/fullFrames -d cuda
    done
done
done

# ---------
# PACK RUNS
# ---------

# Packs some runs for easier distribution
python utils/pack_run.py -b count/game-3 runs/experiment=perineuronal-nets/segmentation/unet_320 pnn_unet_320.zip
python utils/pack_run.py -b count/game-3 runs/experiment=perineuronal-nets/detection/fasterrcnn_640 pnn_fasterrcnn_640.zip

python utils/pack_run.py -b rank/spearman runs_score/method=simple_classification,seed=45/ pnn_scoring_classification.zip
python utils/pack_run.py -b rank/spearman runs_score/method=ordinal_regression,seed=45/ pnn_scoring_ordinal_regression.zip
python utils/pack_run.py -b rank/spearman runs_score/method=pairwise_balanced,seed=45/ pnn_scoring_rank_learning.zip