import h5py
import tifffile

from pathlib import Path
from tqdm import tqdm


def tif_to_h5(tif_path, h5_path):
    print(tif_path, '->', h5_path)

    image = tifffile.imread(tif_path)
    with h5py.File(h5_path, 'w') as h5:
        h5.create_dataset('data', image.shape, image.dtype, data=image, chunks=(1024, 1024))


def main():
    tif_dir = Path('data/perineuronal_nets/fullFrames')
    h5_dir = Path('data/perineuronal_nets/fullFramesH5')

    h5_dir.mkdir(exist_ok=True)
    for tif_path in tqdm(tif_dir.glob('*.tif')):
        h5_path = h5_dir / tif_path.with_suffix('.h5').name
        tif_to_h5(tif_path, h5_path)


if __name__ == '__main__':
    main()