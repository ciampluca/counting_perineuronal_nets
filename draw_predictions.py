import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from skimage import io, transform
from PIL import Image, ImageDraw

RED = np.array([255, 0, 0])
GREEN = np.array([0, 255, 0])

from matplotlib.colors import LinearSegmentedColormap
# cmap = LinearSegmentedColormap.from_list('name', ['red', 'yellow', 'green'])
cmap = LinearSegmentedColormap.from_list('name', [(1,0,0), (1,1,0), (0,1,0)])
# cmap = matplotlib.cm.RdYlGn


def draw_predictions(image, data, radius):

    image = Image.fromarray(image)
    pil_draw = ImageDraw.Draw(image)

    for r, c, s in data[['Y', 'X', 'score']].values:
        if s == -1:
            color = (255, 255, 0)
        else:
            color = 255 * np.array(cmap(s))
            color = tuple(color.astype(int))
        y0, x0 = int(r - radius / 2), int(c - radius / 2)
        y1, x1 = int(r + radius / 2), int(c + radius / 2)
        pil_draw.ellipse([x0, y0, x1, y1], outline=color, width=3)
    
    return np.array(image)


def main(args):
    predictions = pd.read_csv(args.prediction_file)
    predictions[['Y', 'X']] = predictions[['Y', 'X']].values * args.scale_factor

    if 'rescore' in predictions.columns:
        predictions['score'] = predictions['rescore']
    vmin, vmax = predictions.score.values.min(), predictions.score.values.max()
    predictions['score'] = (predictions['score'] - vmin) / (vmax - vmin)

    image_dir = Path(args.root)
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    for image_name, image_predictions in predictions.groupby('imgName'):
        image_path = image_dir / image_name
        if not image_path.exists():
            print(f'Image not found: {image_path}, skipping.')
            continue
        
        image = io.imread(image_path)
        image = transform.rescale(image, args.scale_factor, anti_aliasing=True) if args.scale_factor != 1 else image

        if image.ndim == 2:
            image = (matplotlib.cm.viridis(image)[:,:,:3] * 255).astype(np.uint8)

        out_path = (out_dir / ('clean_' + image_name)).with_suffix('.png')
        print('Saving:', out_path)
        io.imsave(out_path, image)
        
        out_path = (out_dir / ('all_predictions_' + image_name)).with_suffix('.png')
        print('Saving:', out_path)
        drawn = draw_predictions(image, image_predictions, radius=40)
        io.imsave(out_path, drawn)

        out_path = (out_dir / ('low_predictions_' + image_name)).with_suffix('.png')
        print('Saving:', out_path)
        sel = image_predictions.score.between(0, 0.33)
        drawn = draw_predictions(image, image_predictions[sel], radius=40)
        io.imsave(out_path, drawn)

        out_path = (out_dir / ('mid_predictions_' + image_name)).with_suffix('.png')
        print('Saving:', out_path)
        sel = image_predictions.score.between(0.33, 0.66)
        drawn = draw_predictions(image, image_predictions[sel], radius=40)
        io.imsave(out_path, drawn)

        out_path = (out_dir / ('high_predictions_' + image_name)).with_suffix('.png')
        print('Saving:', out_path)
        sel = image_predictions.score.between(0.66, 1)
        drawn = draw_predictions(image, image_predictions[sel], radius=40)
        io.imsave(out_path, drawn)

        out_path = (out_dir / ('loc_' + image_name)).with_suffix('.png')
        print('Saving:', out_path)
        image_predictions = image_predictions.assign(score=-1)
        drawn = draw_predictions(image, image_predictions, radius=40)
        io.imsave(out_path, drawn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw Predictions', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('prediction_file', help='CSV of predictions')
    parser.add_argument('-r', '--root', type=str, default='.', help="Root directory of input images")
    parser.add_argument('-s', '--scale-factor', type=float, default=1, help="Scale factor for image")
    parser.add_argument('-o', '--output', default='.', help="Directory of output drawn images")

    args = parser.parse_args()
    main(args)
