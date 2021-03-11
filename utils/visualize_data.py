import os
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np

from utils.misc import normalize

IMG_NAME = "020_B2_s07_C1"
ROOT_DATA = "/mnt/Dati_SSD_2/datasets/perineural_nets"

RADIUS = 10


if __name__ == '__main__':
    img_path = os.path.join(ROOT_DATA, 'fullFrames', IMG_NAME + ".tif")
    csv_anns_file = os.path.join(ROOT_DATA, 'annotation', 'annotations.csv')
    bbs_path = os.path.join(ROOT_DATA, 'annotation', 'bbs', IMG_NAME + ".txt")
    dmap_path = os.path.join(ROOT_DATA, 'annotation', 'dmaps', IMG_NAME + ".npy")

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    img.save(os.path.join('./output', IMG_NAME + ".png"))

    csv_anns = pd.read_csv(csv_anns_file)
    x_coords = np.array(csv_anns.loc[csv_anns['imageName'].isin(["{}.tif".format(IMG_NAME)]), "X"])
    y_coords = np.array(csv_anns.loc[csv_anns['imageName'].isin(["{}.tif".format(IMG_NAME)]), "Y"])
    img_draw = ImageDraw.Draw(img)
    for x, y in zip(x_coords, y_coords):
        img_draw.ellipse([x-RADIUS, y-RADIUS, x+RADIUS, y+RADIUS], outline='red', width=2)
    img.save(os.path.join('./output', IMG_NAME + "_GTWithDots.png"))

    img = Image.open(img_path).convert("RGB")
    img_draw = ImageDraw.Draw(img)
    with open(bbs_path, 'r') as bounding_box_file:
        for line in bounding_box_file:
            x_center = float(line.split()[0]) * img_w
            y_center = float(line.split()[1]) * img_h
            bb_width = float(line.split()[2]) * img_w
            bb_height = float(line.split()[3]) * img_h
            x_min = x_center - (bb_width / 2.0)
            x_max = x_min + bb_width
            y_min = y_center - (bb_height / 2.0)
            y_max = y_min + bb_height
            img_draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)
    img.save(os.path.join('./output', IMG_NAME + "_GTWithBBs.png"))

    np_dmap = np.load(dmap_path)
    norm_np_dmap = normalize(np_dmap).astype('uint8')
    dmap = Image.fromarray(norm_np_dmap)
    dmap.save(os.path.join('./output', IMG_NAME + "_dmap.png"))
