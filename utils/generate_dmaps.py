import pandas as pd
import numpy as np
import os
from tifffile import imread
import tqdm
from PIL import Image

from utils.get_density_map_gaussian import get_density_map_gaussian
from utils.misc import normalize

ROOT = "/mnt/Dati_SSD_2/datasets/perineural_nets"
KERNEL_SIZE = 61
SIGMA = 30

SAVE = True


if __name__ == '__main__':
    csv_anns_file = os.path.join(ROOT, 'annotation', 'annotations.csv')
    csv_anns = pd.read_csv(csv_anns_file)

    img_names = np.unique(np.array(csv_anns['imageName']))
    img_paths = [ROOT + '/fullFrames/' + img_name for img_name in img_names]

    for img_name in tqdm.tqdm(img_names):
        # Loading image
        img_path = ROOT + '/fullFrames/' + img_name
        img = imread(img_path)

        # Retrieving gt points coords
        x_coords = np.array(csv_anns.loc[csv_anns['imageName'].isin(["{}".format(img_name)]), "X"])
        y_coords = np.array(csv_anns.loc[csv_anns['imageName'].isin(["{}".format(img_name)]), "Y"])

        # Generating dmaps from points
        ann_points = []
        for x, y in zip(x_coords, y_coords):
            ann_points.append([x, y])
        ann_points = np.asarray(ann_points)
        gt_num = ann_points.shape[0]
        dmap = get_density_map_gaussian(img, ann_points, KERNEL_SIZE, SIGMA)

        # Saving generated dmap
        dmap_path = os.path.join(ROOT, 'annotation', 'dmaps', img_name.rsplit(".", 1)[0] + ".npy")
        np.save(dmap_path, dmap)

        # Checking
        stored_dmap = np.load(dmap_path)
        sum_dmap = np.sum(stored_dmap)

        if gt_num != int(round(sum_dmap)):
            print("Different: GT: {}, Den Sum: {}".format(gt_num, int(round(sum_dmap))))

        if SAVE:
            if not os.path.exists("./output/gt/dmaps"):
                os.makedirs("./output/gt/dmaps")
            Image.fromarray(normalize(dmap).astype('uint8')).\
                save(os.path.join("./output/gt/dmaps", img_name.rsplit(".", 1)[0] + ".png"))

