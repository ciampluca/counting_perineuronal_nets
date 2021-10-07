import argparse
from pathlib import Path
import zipfile


def main(args):
    if not args.run.exists():
        print('RUN NOT FOUND:', args.run)
        return
    
    files = [
        '.hydra/config.yaml',
        '.hydra/hydra.yaml',
        '.hydra/overrides.yaml',
        'train_log.csv',
        'valid_log.csv',
    ]

    root = args.output.stem
    with zipfile.ZipFile(args.output, 'w') as archive:
        for file in files:
            archive.write(args.run / file, f'{root}/{file}')
        
        ckpt_path = args.run / 'best_models' / f"best_model_metric_{args.best_metric.replace('/', '-')}.pth"
        archive.write(ckpt_path, f'{root}/best.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Strip a trained run and create a run archive')
    parser.add_argument('run', type=Path, help='path to trained run dir')
    parser.add_argument('output', type=Path, help='output zip archive')
    parser.add_argument('-b', '--best-metric', default='loss', help='which best model to choose')
    args = parser.parse_args()
    main(args)
