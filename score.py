import argparse
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.patched_datasets import RandomAccessMultiImageDataset

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

    patch_size = args.patch_size if args.patch_size else cfg.data.validation.get('patch_size', None)
    dataset_params = dict(
        patch_size=patch_size,
        transforms=hydra.utils.instantiate(cfg.data.validation.transforms),
    )

    localizations = pd.read_csv(args.locs, index_col=0)
    selector = ~localizations.Xp.isna()
    localizations.loc[selector, ['Yi', 'Xi']] = localizations.loc[selector, ['Yp', 'Xp']].round().values

    unique_localizations = localizations.loc[selector, ['imgName', 'Yi', 'Xi']].drop_duplicates().sort_values(['imgName', 'Yi', 'Xi'])

    paths_and_locs = ((Path(args.root) / image, data[['Yi', 'Xi']].values.astype(int)) for image, data in unique_localizations.groupby('imgName'))
    paths, locs = zip(*paths_and_locs)

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

    metric_name = args.metric
    ckpt_path = run_path / 'best.pth'
    if not ckpt_path.exists():
        ckpt_path = run_path / 'best_models' / f"best_model_metric_{metric_name.replace('/', '-')}.pth"
    if not ckpt_path.exists():
        ckpt_path = run_path / 'last.pth'

    print(f"[  CKPT] {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    scores = score_patches(loader, model, device, cfg)

    unique_localizations['rescore'] = scores
    merged = localizations.merge(unique_localizations, on=['imgName', 'Yi', 'Xi'], how='left')

    print(f'[OUTPUT] {args.output}')
    merged.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Counting and Localization', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run', help='Pretrained run directory')
    parser.add_argument('locs', help='CSV of images/patches locations to rescore')
    parser.add_argument('-m', '--metric', type=str, default='loss', help="Metric on which select checkpoint to use")
    parser.add_argument('-r', '--root', type=str, default='.', help="Root directory of input images")
    parser.add_argument('-d', '--device', default='cpu', help="Device to use; e.g., 'cpu', 'cuda:0'")
    parser.add_argument('-p', '--patch-size', type=int, help="Patch size (side of squared region around localization point)")
    parser.add_argument('-b', '--batch-size', type=int, default=512, help="Batch size (number of patches processed in parallel by the model)")
    parser.add_argument('-o', '--output', default='rescored_predictions.csv', help="Output file")

    args = parser.parse_args()
    main(args)
