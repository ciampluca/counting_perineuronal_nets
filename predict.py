import argparse
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.patched_datasets import PatchedMultiImageDataset, RandomAccessMultiImageDataset

@torch.no_grad()
def score_patches(loader, model, device, cfg):
    compute_loss_and_scores = hydra.utils.get_method(f'methods.rank.methods.{cfg.optim.method}')

    model.eval()
    all_scores = []
    for sample in tqdm(loader, desc='PRED', leave=False, dynamic_ncols=True):
        dummy_targets = torch.zeros(sample.shape[0], dtype=torch.int64, device=device)
        sample = (sample, dummy_targets)
        batch_metrics, scores = compute_loss_and_scores(sample, model, device, cfg)
        all_scores.append(scores.flatten().cpu())
    
    scores = torch.cat(all_scores).numpy()
    return scores


def main(args):
    run_path = Path(args.run)
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
    ckpt_path = run_path / 'best.pth'
    if not ckpt_path.exists():
        ckpt_path = run_path / 'best_models' / f"best_model_metric_{metric_name.replace('/', '-')}.pth"
    print(f"[  CKPT] {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    threshold = checkpoint['metrics'][metric_name]['threshold'] if args.threshold is None else args.threshold
    print(f'[PARAMS] thr = {threshold:.2f}')

    predict_points = hydra.utils.get_method(f'methods.{cfg.method}.train_fn.predict_points')
    localizations = predict_points(loader, model, device, threshold, cfg)
    localizations = localizations.sort_values(['imgName', 'Y', 'X'])

    print(f'[OUTPUT] {args.output}')
    localizations.to_csv(args.output, index=False)

    if args.rescore:
        run_path = Path(args.rescore)
        cfg_path = run_path / '.hydra' / 'config.yaml'
        cfg = OmegaConf.load(cfg_path)
        cfg['cache_folder'] = './model_zoo'

        dataset_params = dict(
            patch_size=cfg.data.validation.get('patch_size', None),
            transforms=hydra.utils.instantiate(cfg.data.validation.transforms),
        )

        paths_and_locs = ((image, data[['Y', 'X']].values.astype(int)) for image, data in localizations.groupby('imgName'))
        paths, locs = zip(*paths_and_locs)
        paths = [Path(args.data[0]).parent / p for p in paths]  # TODO ugly hack

        dataset = RandomAccessMultiImageDataset.from_paths_and_locs(paths, locs, **dataset_params)
        print(f'[  DATA] {dataset}')

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        model_params = cfg.model.get('wrapper', cfg.model.base)
        model = hydra.utils.instantiate(model_params)
        model_param_string = ', '.join(f'{k}={v}' for k, v in model_params.items() if not k.startswith('_'))
        print(f"[ MODEL] {cfg.model.name}({model_param_string})")

        device = torch.device(args.device)
        model = model.to(device)
        print(f'[DEVICE] {device}')

        metric_name = 'rank/spearman'
        ckpt_path = run_path / 'best.pth'
        if not ckpt_path.exists():
            ckpt_path = run_path / 'best_models' / f"best_model_metric_{metric_name.replace('/', '-')}.pth"
        if not ckpt_path.exists():
            ckpt_path = run_path / 'last.pth'

        print(f"[  CKPT] {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

        scores = score_patches(loader, model, device, cfg)
        localizations['rescore'] = scores

        print(f'[OUTPUT] {args.output}')
        localizations.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Counting and Localization', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run', help='Pretrained run directory')
    parser.add_argument('data', nargs='+', help='Input Images (Image or HDF5 formats)')
    parser.add_argument('-d', '--device', default='cpu', help="Device to use; e.g., 'cpu', 'cuda:0'")
    parser.add_argument('-b', '--batch-size', type=int, default=1, help="Batch size (number of patches processed in parallel by the model)")
    parser.add_argument('-r', '--rescore', type=str, default=None, help="Pretrain run directory of rescoring model")
    parser.add_argument('-t', '--threshold', type=float, default=None, help="Threshold (good values may vary depending on the model)")
    parser.add_argument('-o', '--output', default='localizations.csv', help="Output file")    

    args = parser.parse_args()
    main(args)
