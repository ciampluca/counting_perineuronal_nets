import pandas as pd
import numpy as np
import os
import albumentations as A
import albumentations.augmentations.bbox_utils as albumentations_utils
from tifffile import imread, imwrite
import tqdm
import csv
from PIL import Image, ImageDraw


ROOT = "/mnt/Dati_SSD_2/datasets/perineural_nets"
SRC_IMGS = "{}/fullFrames/".format(ROOT)

BB_W = 60
BB_H = 60

DST_FRAMES = "{}/specular_fullFrames".format(ROOT)
DST_ANN_CSV_FILE = "{}/annotation/specular_annotations.csv".format(ROOT)
DST_BB_ANNS = "{}/annotation/specular_bbs".format(ROOT)

SAVE = True


if __name__ == "__main__":
    csv_anns_file = os.path.join(ROOT, 'annotation', 'annotations.csv')
    csv_anns = pd.read_csv(csv_anns_file)

    img_names = np.unique(np.array(csv_anns['imageName']))
    img_paths = [SRC_IMGS + img_name for img_name in img_names]

    names, ordinate, abscissa = [], [], []
    for img_name in tqdm.tqdm(img_names):
        # Loading image
        img_path = ROOT + '/fullFrames/' + img_name
        img = imread(img_path)
        img_h, img_w = img.shape[:2]

        img_left_w = int(img_w / 2)
        img_left_region = img[0:img_h, 0:img_left_w]
        img_right_region = img[0:img_h, img_left_w+1:img_w]
        img_left_h, img_left_w = img_left_region.shape[:2]
        img_right_h, img_right_w = img_right_region.shape[:2]

        # Retrieving gt points coords
        x_coords = np.array(csv_anns.loc[csv_anns['imageName'].isin(["{}".format(img_name)]), "X"])
        y_coords = np.array(csv_anns.loc[csv_anns['imageName'].isin(["{}".format(img_name)]), "Y"])
        img_ann_points = []
        img_left_ann_points, img_right_ann_points = [], []
        for x, y in zip(x_coords, y_coords):
            if x <= img_left_w:
                img_left_ann_points.append((x, y))
            else:
                x = x-(img_left_w+1)
                if x < 0:
                    x = 0
                img_right_ann_points.append((x, y))

        ################################
        # Left region
        img_left_name = "{}_left.tif".format(img_name.rsplit(".", 1)[0])
        imwrite(os.path.join(DST_FRAMES, img_left_name), img_left_region)

        names.extend([img_left_name] * len(img_left_ann_points))
        ordinate.extend([point[0] for point in img_left_ann_points])
        abscissa.extend([point[1] for point in img_left_ann_points])

        # Computing bbs from points
        img_left_ann_bbs = []
        for point in img_left_ann_points:
            x_min = point[0] - (BB_W / 2.0)
            x_max = x_min + BB_W
            y_min = point[1] - (BB_H / 2.0)
            y_max = y_min + BB_H
            img_left_ann_bbs.append([x_min, y_min, x_max, y_max])

        # Clipping
        img_left_ann_bbs = [np.clip(np.asarray(bb), [0, 0, 0, 0], [img_left_w, img_left_h, img_left_w, img_left_h]).tolist()
                             for bb in img_left_ann_bbs]
        img_left_ann_bbs = [tuple(bb) for bb in img_left_ann_bbs]

        # Converting to albumentations format and checking validity
        img_left_ann_bbs_alb_format = albumentations_utils.convert_bboxes_to_albumentations(
            bboxes=img_left_ann_bbs,
            source_format='pascal_voc',
            rows=img_left_h,
            cols=img_left_w,
            check_validity=True,
        )

        # Converting to yolo format and checking validity
        img_left_ann_bbs_yolo_format = albumentations_utils.convert_bboxes_from_albumentations(
            bboxes=img_left_ann_bbs_alb_format,
            target_format='yolo',
            rows=img_left_h,
            cols=img_left_w,
            check_validity=True,
        )

        # Saving bbs coords in a txt file
        bb_txt_file_path = os.path.join(DST_BB_ANNS, img_left_name.rsplit(".", 1)[0] + ".txt")
        img_left_ann_bbs_yolo_format = [list(elem) for elem in img_left_ann_bbs_yolo_format]
        with open(bb_txt_file_path, 'w') as f:
            csv_writer = csv.writer(f, delimiter=' ')
            csv_writer.writerows(img_left_ann_bbs_yolo_format)

        if SAVE:
            pil_img = Image.fromarray(img_left_region).convert('RGB')
            image_draw = ImageDraw.Draw(pil_img)
            if not os.path.exists("./output/gt/bbs"):
                os.makedirs("./output/gt/bbs")
            for bb in img_left_ann_bbs_yolo_format:
                x_center = bb[0] * img_left_w
                y_center = bb[1] * img_left_h
                bb_width = bb[2] * img_left_w
                bb_height = bb[3] * img_left_h
                x_min = x_center - (bb_width / 2.0)
                x_max = x_min + bb_width
                y_min = y_center - (bb_height / 2.0)
                y_max = y_min + bb_height
                image_draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
            pil_img.save(os.path.join("./output/gt/bbs", img_left_name))

        ################################
        # Right region
        img_right_name = "{}_right.tif".format(img_name.rsplit(".", 1)[0])
        imwrite(os.path.join(DST_FRAMES, img_right_name), img_right_region)

        names.extend([img_right_name] * len(img_right_ann_points))
        ordinate.extend([point[0] for point in img_right_ann_points])
        abscissa.extend([point[1] for point in img_right_ann_points])

        # Computing bbs from points
        img_right_ann_bbs = []
        for point in img_right_ann_points:
            x_min = point[0] - (BB_W / 2.0)
            x_max = x_min + BB_W
            y_min = point[1] - (BB_H / 2.0)
            y_max = y_min + BB_H
            img_right_ann_bbs.append([x_min, y_min, x_max, y_max])

        # Clipping
        img_right_ann_bbs = [np.clip(np.asarray(bb), [0, 0, 0, 0], [img_right_w, img_right_h, img_right_w, img_right_h]).tolist()
                             for bb in img_right_ann_bbs]
        img_right_ann_bbs = [tuple(bb) for bb in img_right_ann_bbs]

        # Converting to albumentations format and checking validity
        img_right_ann_bbs_alb_format = albumentations_utils.convert_bboxes_to_albumentations(
            bboxes=img_right_ann_bbs,
            source_format='pascal_voc',
            rows=img_right_h,
            cols=img_right_w,
            check_validity=True,
        )

        # Converting to yolo format and checking validity
        img_right_ann_bbs_yolo_format = albumentations_utils.convert_bboxes_from_albumentations(
            bboxes=img_right_ann_bbs_alb_format,
            target_format='yolo',
            rows=img_right_h,
            cols=img_right_w,
            check_validity=True,
        )

        # Saving bbs coords in a txt file
        bb_txt_file_path = os.path.join(DST_BB_ANNS, img_right_name.rsplit(".", 1)[0] + ".txt")
        img_right_ann_bbs_yolo_format = [list(elem) for elem in img_right_ann_bbs_yolo_format]
        with open(bb_txt_file_path, 'w') as f:
            csv_writer = csv.writer(f, delimiter=' ')
            csv_writer.writerows(img_right_ann_bbs_yolo_format)

        if SAVE:
            pil_img = Image.fromarray(img_right_region).convert('RGB')
            image_draw = ImageDraw.Draw(pil_img)
            if not os.path.exists("./output/gt/bbs"):
                os.makedirs("./output/gt/bbs")
            for bb in img_right_ann_bbs_yolo_format:
                x_center = bb[0] * img_right_w
                y_center = bb[1] * img_right_h
                bb_width = bb[2] * img_right_w
                bb_height = bb[3] * img_right_h
                x_min = x_center - (bb_width / 2.0)
                x_max = x_min + bb_width
                y_min = y_center - (bb_height / 2.0)
                y_max = y_min + bb_height
                image_draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
            pil_img.save(os.path.join("./output/gt/bbs", img_right_name))

    df = pd.DataFrame({'imageName': names, 'X': ordinate, 'Y': abscissa})
    df.to_csv(DST_ANN_CSV_FILE, index=False)

    print("Exiting...")
