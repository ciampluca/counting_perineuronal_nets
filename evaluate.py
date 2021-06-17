# -*- coding: utf-8 -*-
import argparse
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def main(args):
    run_path = Path(args.run)
    hydra_cfg = OmegaConf.load(run_path / '.hydra' / 'hydra.yaml')['hydra']
    OmegaConf.register_resolver("hydra", lambda x: OmegaConf.select(hydra_cfg, x))

    cfg = OmegaConf.load(run_path / '.hydra' / 'config.yaml')
    print(OmegaConf.to_yaml(cfg))

    cfg['cache_folder'] = './model_zoo'

    device = torch.device(args.device)

    # create test dataset and dataloader
    test_dataset = cfg.data.validation
    test_dataset.root = args.data_root if args.data_root else test_dataset.root
    test_dataset.split = 'test'
    test_dataset.target_ = None

    print(OmegaConf.to_yaml(test_dataset))

    test_dataset = hydra.utils.instantiate(test_dataset)



    test_batch_size = cfg.optim.batch_size
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=cfg.optim.num_workers)
    log.info(f'[TEST] {test_dataset}')

    # create model and move to device
    model = hydra.utils.instantiate(cfg.model.module, skip_weights_loading=True)
    model.to(device)

    # resume from a saved checkpoint
    best_models_folder = run_path / 'best_models'
    metric_name = args.best_on_metric.replace('/', '-')
    ckpt_path = best_models_folder / f'best_model_metric_{metric_name}.pth'
    log.info(f"[CKPT]: Loading {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    outdir = (run_path / 'test_predictions') if args.save else None
    predict = hydra.utils.get_method(f'{cfg.method}.train_fn.predict')
    predict(test_loader, model, device, cfg, outdir, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('run', help='Path to run dir')
    parser.add_argument('-d', '--device', default='cuda', help='device to use for prediction')
    parser.add_argument('--best-on-metric', default='count/game-3', help='select snapshot that optimizes this metric')
    parser.add_argument('--no-save', action='store_false', dest='save', help='draw images with predictions')
    parser.add_argument('--debug', action='store_true', default=False, help='draw images with predictions')
    parser.add_argument('--data-root', default=None, help='root of the test subset')
    parser.set_defaults(save=True)

    args = parser.parse_args()
    main(args)
