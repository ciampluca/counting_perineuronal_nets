import argparse
import itertools
from pathlib import Path
import shutil

import torch


def convert_ckpt(src, dest):

    if not src.exists():
        print(f'SKIPPING NON-EXISTING: {src.name}')
        return

    if dest.exists():
        print(f'SKIPPING CKPT: {src.name} -> {dest}')
        return

    ckpt = torch.load(src, map_location='cpu')
    values = ckpt.pop('best_validation_metrics')
    thresholds = ckpt.pop('best_thresholds')
    epochs = ckpt.pop('best_metrics_epoch')

    ckpt['best_metrics'] = {
        metric: {
            'value': value,
            'threshold': thresholds.get(metric, None),
            'epoch': epochs.get(metric, None)
        } for metric, value in values.items()
    }

    torch.save(ckpt, dest)


def main(args):
    assert args.run.exists(), f'Run not found {args.run}'
    args.destination.mkdir(parents=True, exist_ok=True)

    print(' == Copying logs ...')
    logs = itertools.chain(
        args.run.glob('*.csv'),
        args.run.glob('*.log'),
        [args.run / 'runs']
    )
    for src in logs:
        dest = args.destination / src.name
        if dest.exists():
            print(f'SKIPPING EXISTING: {src.name} -> {dest}')
            continue

        print(f'{src.name} -> {dest}')
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy(src, dest)
    
    print(' == Copying checkpoints ...')
    src = args.run / 'last.pth'
    dest = args.destination / 'last.pth'
    convert_ckpt(src, dest)

    src_ckpt_dir = args.run / 'best_models/'
    dest_ckpt_dir = args.destination / 'best_models/'
    dest_ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_file in src_ckpt_dir.glob('*.pth'):
        
        dest_link_ckpt = dest_ckpt_dir / ckpt_file.name.replace('_VGGCellsDataset', '').replace('_perineural_nets', '').replace('_best.pth', '.pth')
        if dest_link_ckpt.exists():
            print(f'SKIPPING CKPT: {dest_link_ckpt}')
            continue

        print(f'LOADING CKPT: {ckpt_file}')
        ckpt = torch.load(ckpt_file, map_location='cpu')
        epoch = ckpt['epoch']
        metrics = ckpt.pop('metrics')
        ckpt['metrics'] = {
            k.replace('_best', ''): {
                'value': metrics[k],
                'threshold': metrics.get(f'{k}_thr', None),
                'epoch': epoch
            } for k in metrics.keys() if not k.endswith('_thr')
        }

        dest_ckpt = dest_ckpt_dir / f'ckpt_e{epoch}.pth'
        if not dest_ckpt.exists():
            print(f'{ckpt_file} -> {dest_ckpt}')
            torch.save(ckpt, dest_ckpt)
        else:
            print(f'SKIPPING EXISTING: {dest_ckpt}')
        
        print(f'LINKING CKPT: {ckpt_file} -> {dest_link_ckpt} [{dest_ckpt.name}]')
        dest_link_ckpt.symlink_to(dest_ckpt.name)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts v0.1 trained runs into v0.2 runs')
    parser.add_argument('run', type=Path, help='Path to old run dir')
    parser.add_argument('destination', type=Path, help='Path to new run dir')
    args = parser.parse_args()
    main(args)