# -*- coding: utf-8 -*-
import argparse
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

@torch.no_grad()
def predict(loader, model, device, cfg, outdir, debug=False):
    compute_loss_and_scores = hydra.utils.get_method(f'methods.rank.methods.{cfg.optim.method}')

    model.eval()
    all_scores = []
    for sample in tqdm(loader, desc='PRED', leave=False, dynamic_ncols=True):
        batch_metrics, scores = compute_loss_and_scores(sample, model, device, cfg)
        all_scores.append(scores.flatten().cpu())
    
    scores = torch.cat(all_scores).numpy()
    all_gt_preds = loader.dataset.annot.copy()
    all_gt_preds['score'] = scores

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        all_gt_preds.to_csv(outdir / 'all_gt_preds.csv.gz')
    
    corr_coef, _ = spearmanr(all_gt_preds.agreement, all_gt_preds.score)
    print(f'Spearman: {corr_coef}')
    #breakpoint()


def main(args):
    run_path = Path(args.run)

    hydra_cfg = OmegaConf.load(run_path / '.hydra' / 'hydra.yaml')['hydra']
    OmegaConf.register_new_resolver("hydra", lambda x: OmegaConf.select(hydra_cfg, x))

    cfg = OmegaConf.load(run_path / '.hydra' / 'config.yaml')
    #print(OmegaConf.to_yaml(cfg))

    device = torch.device(args.device)

    # create test dataset and dataloader
    test_dataset = cfg.data.validation
    test_dataset.root = args.data_root if args.data_root else test_dataset.root
    test_dataset.split = 'test'
    test_dataset.mode = 'patches'
    test_dataset.neg_fraction = 0

    test_dataset = hydra.utils.instantiate(test_dataset)

    test_batch_size = cfg.optim.batch_size
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=cfg.optim.num_workers)
    log.info(f'[TEST] {test_dataset}')

    # create model and move to device
    model_params = cfg.model.get('wrapper', cfg.model.base)
    model = hydra.utils.instantiate(model_params).to(device)
    model_param_string = ', '.join(f'{k}={v}' for k, v in model_params.items() if not k.startswith('_'))
    log.info(f"[MODEL] {cfg.model.name}({model_param_string})")

    # resume from a saved checkpoint
    best_models_folder = run_path / 'best_models'
    metric_name = args.best_on_metric.replace('/', '-')
    ckpt_path = best_models_folder / f'best_model_metric_{metric_name}.pth'
    if not ckpt_path.exists():
        ckpt_path = run_path / 'last.pth'
    log.info(f"[CKPT]: Loading {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    outdir = (run_path / 'test_predictions') if args.save else None
    # predict = hydra.utils.get_method(f'methods.{cfg.method}.train_fn.predict')
    predict(test_loader, model, device, cfg, outdir, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform evaluation on test set')
    parser.add_argument('run', help='Path to run dir')
    parser.add_argument('-d', '--device', default='cuda', help='device to use for prediction')
    parser.add_argument('--best-on-metric', default='loss', help='select snapshot that optimizes this metric')
    parser.add_argument('--no-save', action='store_false', dest='save', help='draw images with predictions')
    parser.add_argument('--debug', action='store_true', default=False, help='draw images with predictions')
    parser.add_argument('--data-root', default=None, help='root of the test subset')
    parser.set_defaults(save=True)

    args = parser.parse_args()
    main(args)
