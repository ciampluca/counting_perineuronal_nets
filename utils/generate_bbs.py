import pandas as pd
import numpy as np
import os
from tifffile import imread
import csv
import tqdm
from PIL import Image, ImageDraw
import albumentations.augmentations.bbox_utils as albumentations_utils


ROOT = "/mnt/Dati_SSD_2/datasets/perineural_nets"
BB_W = 60
BB_H = 60

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
        img_h, img_w = img.shape[:2]

        # Retrieving gt points coords
        x_coords = np.array(csv_anns.loc[csv_anns['imageName'].isin(["{}".format(img_name)]), "X"])
        y_coords = np.array(csv_anns.loc[csv_anns['imageName'].isin(["{}".format(img_name)]), "Y"])

        # Computing bbs from points
        bbs = []
        for x, y in zip(x_coords, y_coords):
            x_min = x - (BB_W / 2.0)
            x_max = x_min + BB_W
            y_min = y - (BB_H / 2.0)
            y_max = y_min + BB_H
            bbs.append([x_min, y_min, x_max, y_max])

        # Clipping
        bbs = [np.clip(np.asarray(bb), [0, 0, 0, 0], [img_w, img_h, img_w, img_h]).tolist()
                             for bb in bbs]
        bbs = [tuple(bb) for bb in bbs]

        # Converting to albumentations format and checking validity
        bbs_alb_format = albumentations_utils.convert_bboxes_to_albumentations(
            bboxes=bbs,
            source_format='pascal_voc',
            rows=img_h,
            cols=img_w,
            check_validity=True,
        )

        # Converting to yolo format and checking validity
        bbs_yolo_format = albumentations_utils.convert_bboxes_from_albumentations(
            bboxes=bbs_alb_format,
            target_format='yolo',
            rows=img_h,
            cols=img_w,
            check_validity=True,
        )

        # Saving bbs coords in a txt file
        bb_txt_file_path = os.path.join(ROOT, 'annotation', 'bbs', img_name.rsplit(".", 1)[0] + ".txt")
        with open(bb_txt_file_path, 'w') as f:
            csv_writer = csv.writer(f, delimiter=' ')
            csv_writer.writerows(bbs_yolo_format)

        if SAVE:
            img = Image.open(img_path).convert("RGB")
            image_draw = ImageDraw.Draw(img)
            if not os.path.exists("./output/gt/bbs"):
                os.makedirs("./output/gt/bbs")
            for x, y in zip(x_coords, y_coords):
                x_tl = x - (BB_W/2)
                y_tl = y - (BB_H / 2)
                x_br = x_tl + BB_W
                y_br = y_tl + BB_H
                image_draw.rectangle([x_tl, y_tl, x_br, y_br], outline='red', width=2)
            img.save(os.path.join("./output/gt/bbs", img_name.rsplit(".", 1)[0] + ".png"))



