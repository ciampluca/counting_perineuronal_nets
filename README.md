# Counting Perineuronal Nets

PyTorch code for training and evaluating cell counting and localization methodologies.

## Getting Started

- Download the data in `data/` (TODO)
- Install requirements (see [`Dockerfile`](Dockerfile))

## How to train

Train configurations are specified with [Hydra](https://hydra.cc/) config groups in `conf/experiments`. You can run a training experiment by passing as argument `experiment=<exp_name>` to the `train.py` script, where `<exp_name>` is the path to a YAML experiment configuration relative to `conf/experiments` and without the `.yaml` extension.

### Examples:

- Train the detection-based approach (FasterRCNN) with 480x480 patches on PerineuronalNets:
  ```bash
  python train.py experiment=perineuronal-nets/detection/fasterrcnn_480
  ```

- Train the density-based approach (CSRNet) on VGG Cells:
  ```bash
  python train.py experiment=vgg-cells/density/csrnet
  ```

Runs files will be produced in the `runs` folder. Once trained, you can evaluate the trained models on the corresponding test sets using the `evaluate.py` script.
E.g.,
```bash
# check python evaluate.py -h for more options
python evaluate.py runs/experiment=perineuronal-nets/detection/fasterrcnn_480
```
Metrics and predictions will be saved in the run folder under `test_predictions/`.

## How to do predictions

To do predictions, you need a run folder containing a pretrained model. Then, you can do predictions using the `predict.py` script by passing the run folder and the paths to data to process. E.g.:

```bash
# check python predict.py -h for more options
python predict.py runs/experiment=perineuronal-nets/detection/fasterrcnn_480 my_dataset/*.tiff
```

Accepted format for input data are image formats and the HDF5 format. For HDF5, we assume there is a `data` dataset containing a sigle 1-channel (bidimensional) image.
