import argparse
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from datasets.patched_datasets import PatchedMultiImageDataset


def main(args):
    run_path = Path(args.model)
    cfg_path = run_path / '.hydra' / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)
    cfg['cache_folder'] = './model_zoo'

    dataset_params = dict(
        patch_size=cfg.data.validation.get('patch_size', None),
        transforms=hydra.utils.instantiate(cfg.data.validation.transforms),
    )
    dataset = PatchedMultiImageDataset.from_paths(args.data, **dataset_params)
    print(f'[  DATA] {dataset}')

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model_param_string = ', '.join(f'{k}={v}' for k, v in cfg.model.module.items() if not k.startswith('_'))
    model = hydra.utils.instantiate(cfg.model.module, skip_weights_loading=True)
    print(f"[ MODEL] {cfg.method} - {cfg.model.name}({model_param_string})")

    device = torch.device(args.device)
    model = model.to(device)
    print(f'[DEVICE] {device}')

    metric_name = 'count/game-3'
    ckpt_path = run_path / 'best_models' / f"best_model_metric_{metric_name.replace('/', '-')}.pth"
    print(f"[  CKPT] {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    threshold = checkpoint['metrics'][metric_name]['threshold'] if args.threshold is None else args.threshold
    print(f'[PARAMS] thr = {threshold:.2f}')

    predict_points = hydra.utils.get_method(f'methods.{cfg.method}.train_fn.predict_points')
    localizations = predict_points(loader, model, device, threshold, cfg)
    print(f'[OUTPUT] {args.output}')
    localizations.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Counting and Localization', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', help='Model Name')
    parser.add_argument('data', nargs='+', help='Input Images (Image or HDF5 formats)')
    parser.add_argument('-d', '--device', default='cpu', help="Device to use; e.g., 'cpu', 'cuda:0'")
    parser.add_argument('-b', '--batch-size', type=int, default=1, help="Device to use; e.g., 'cpu', 'cuda:0'")
    parser.add_argument('-t', '--threshold', type=float, default=None, help="Threshold (good values may vary depending on the model)")
    parser.add_argument('-o', '--output', default='localizations.csv', help="Output file")    

    args = parser.parse_args()
    main(args)
