import pandas as pd
import numpy as np
import albumentations as A
import albumentations.augmentations.bbox_utils as albumentations_utils
from tifffile import imread, imwrite
import tqdm
import csv
from PIL import Image, ImageDraw
import os
from utils.misc import normalize


ROOT = "/mnt/Dati_SSD_2/datasets/perineural_nets"
SRC_IMGS = "{}/fullFrames/".format(ROOT)
SRC_DMAPS = os.path.join(ROOT, 'annotation', 'dmaps')

NUM_PATCHES_PER_IMAGE = 250
PATCH_WIDTH, PATCH_HEIGHT = 1024, 1024
BB_W = 60
BB_H = 60

DST_PATCHES = "{}/specular_patches".format(ROOT)
DST_ANN_CSV_FILE = "{}/annotation/specular_patches_annotations.csv".format(ROOT)
DST_BB_ANNS = "{}/annotation/specular_patches_bbs".format(ROOT)
DST_DMAPS = os.path.join(ROOT, 'annotation', 'specular_patches_dmaps')

SAVE = True


if __name__ == "__main__":
    csv_anns_file = os.path.join(ROOT, 'annotation', 'annotations.csv')
    csv_anns = pd.read_csv(csv_anns_file)

    img_names = np.unique(np.array(csv_anns['imageName']))
    img_paths = [os.path.join(SRC_IMGS, img_name) for img_name in img_names]

    transform = A.Compose([
        A.RandomCrop(width=PATCH_WIDTH, height=PATCH_HEIGHT)],
        keypoint_params=A.KeypointParams(format='xy'),
        additional_targets={'dmap': 'image'}
    )

    names, ordinate, abscissa = [], [], []
    for img_name in tqdm.tqdm(img_names):
        # Loading image and dmap
        img_path = os.path.join(SRC_IMGS, img_name)
        img = imread(img_path)
        img_h, img_w = img.shape[:2]
        dmap = np.load(os.path.join(SRC_DMAPS, img_name.rsplit(".", 1)[0] + ".npy"))

        img_left_w = int(img_w / 2)
        img_left_region = img[0:img_h, 0:img_left_w]
        img_right_region = img[0:img_h, img_left_w+1:img_w]
        dmap_left_region = dmap[0:img_h, 0:img_left_w]
        dmap_right_region = dmap[0:img_h, img_left_w+1:img_w]

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

        for i in range(NUM_PATCHES_PER_IMAGE):
            ################################
            # Left region
            img_patch_name = "{}_left_{}.tif".format(img_name.rsplit(".", 1)[0], i)

            transformed = transform(image=img_left_region, keypoints=img_left_ann_points, dmap=dmap_left_region)

            img_left_patch, img_left_patch_ann_points, dmap_left_patch = \
                transformed['image'], transformed['keypoints'], transformed['dmap']

            imwrite(os.path.join(DST_PATCHES, img_patch_name), img_left_patch)
            np.save(os.path.join(DST_DMAPS, img_patch_name.rsplit(".", 1)[0] + ".npy"), dmap_left_patch)

            names.extend([img_patch_name] * len(img_left_patch_ann_points))
            ordinate.extend([point[0] for point in img_left_patch_ann_points])
            abscissa.extend([point[1] for point in img_left_patch_ann_points])

            # Computing bbs from points
            img_left_patch_ann_bbs = []
            for point in img_left_patch_ann_points:
                x_min = point[0] - (BB_W / 2.0)
                x_max = x_min + BB_W
                y_min = point[1] - (BB_H / 2.0)
                y_max = y_min + BB_H
                img_left_patch_ann_bbs.append([x_min, y_min, x_max, y_max])

            # Clipping
            img_left_patch_ann_bbs = [np.clip(np.asarray(bb), [0, 0, 0, 0], [PATCH_WIDTH, PATCH_HEIGHT, PATCH_WIDTH, PATCH_HEIGHT]).tolist()
                                 for bb in img_left_patch_ann_bbs]
            img_left_patch_ann_bbs = [tuple(bb) for bb in img_left_patch_ann_bbs]

            # Converting to albumentations format and checking validity
            img_left_patch_ann_bbs_alb_format = albumentations_utils.convert_bboxes_to_albumentations(
                bboxes=img_left_patch_ann_bbs,
                source_format='pascal_voc',
                rows=PATCH_HEIGHT,
                cols=PATCH_WIDTH,
                check_validity=True,
            )

            # Converting to yolo format and checking validity
            img_left_patch_ann_bbs_yolo_format = albumentations_utils.convert_bboxes_from_albumentations(
                bboxes=img_left_patch_ann_bbs_alb_format,
                target_format='yolo',
                rows=PATCH_HEIGHT,
                cols=PATCH_WIDTH,
                check_validity=True,
            )

            # Saving bbs coords in a txt file
            bb_txt_file_path = os.path.join(DST_BB_ANNS, img_patch_name.rsplit(".", 1)[0] + ".txt")
            img_left_patch_ann_bbs_yolo_format = [list(elem) for elem in img_left_patch_ann_bbs_yolo_format]
            with open(bb_txt_file_path, 'w') as f:
                csv_writer = csv.writer(f, delimiter=' ')
                csv_writer.writerows(img_left_patch_ann_bbs_yolo_format)

            if SAVE:
                pil_img = Image.fromarray(img_left_patch).convert('RGB')
                image_draw = ImageDraw.Draw(pil_img)
                if not os.path.exists("./output/gt/bbs_patches"):
                    os.makedirs("./output/gt/bbs_patches")
                for bb in img_left_patch_ann_bbs_yolo_format:
                    x_center = bb[0] * PATCH_WIDTH
                    y_center = bb[1] * PATCH_HEIGHT
                    bb_width = bb[2] * PATCH_WIDTH
                    bb_height = bb[3] * PATCH_HEIGHT
                    x_min = x_center - (bb_width / 2.0)
                    x_max = x_min + bb_width
                    y_min = y_center - (bb_height / 2.0)
                    y_max = y_min + bb_height
                    image_draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
                pil_img.save(os.path.join("./output/gt/bbs_patches", img_patch_name))
                if not os.path.exists("./output/gt/dmaps"):
                    os.makedirs("./output/gt/dmaps")
                Image.fromarray(normalize(dmap_left_patch).astype('uint8')). \
                    save(os.path.join("./output/gt/dmaps", img_patch_name.rsplit(".", 1)[0] + ".png"))

            ################################
            # Right region
            img_patch_name = "{}_right_{}.tif".format(img_name.rsplit(".", 1)[0], i)

            transformed = transform(image=img_right_region, keypoints=img_right_ann_points, dmap=dmap_right_region)

            img_right_patch, img_right_patch_ann_points, dmap_right_patch = \
                transformed['image'], transformed['keypoints'], transformed['dmap']

            imwrite(os.path.join(DST_PATCHES, img_patch_name), img_right_patch)
            np.save(os.path.join(DST_DMAPS, img_patch_name.rsplit(".", 1)[0] + ".npy"), dmap_right_patch)

            names.extend([img_patch_name] * len(img_right_patch_ann_points))
            ordinate.extend([point[0] for point in img_right_patch_ann_points])
            abscissa.extend([point[1] for point in img_right_patch_ann_points])

            # Computing bbs from points
            img_right_patch_ann_bbs = []
            for point in img_right_patch_ann_points:
                x_min = point[0] - (BB_W / 2.0)
                x_max = x_min + BB_W
                y_min = point[1] - (BB_H / 2.0)
                y_max = y_min + BB_H
                img_right_patch_ann_bbs.append([x_min, y_min, x_max, y_max])

            # Clipping
            img_right_patch_ann_bbs = [
                np.clip(np.asarray(bb), [0, 0, 0, 0], [PATCH_WIDTH, PATCH_HEIGHT, PATCH_WIDTH, PATCH_HEIGHT]).tolist()
                for bb in img_right_patch_ann_bbs]
            img_right_patch_ann_bbs = [tuple(bb) for bb in img_right_patch_ann_bbs]

            # Converting to albumentations format and checking validity
            img_right_patch_ann_bbs_alb_format = albumentations_utils.convert_bboxes_to_albumentations(
                bboxes=img_right_patch_ann_bbs,
                source_format='pascal_voc',
                rows=PATCH_HEIGHT,
                cols=PATCH_WIDTH,
                check_validity=True,
            )

            # Converting to yolo format and checking validity
            img_right_patch_ann_bbs_yolo_format = albumentations_utils.convert_bboxes_from_albumentations(
                bboxes=img_right_patch_ann_bbs_alb_format,
                target_format='yolo',
                rows=PATCH_HEIGHT,
                cols=PATCH_WIDTH,
                check_validity=True,
            )

            # Saving bbs coords in a txt file
            bb_txt_file_path = os.path.join(DST_BB_ANNS, img_patch_name.rsplit(".", 1)[0] + ".txt")
            img_right_patch_ann_bbs_yolo_format = [list(elem) for elem in img_right_patch_ann_bbs_yolo_format]
            with open(bb_txt_file_path, 'w') as f:
                csv_writer = csv.writer(f, delimiter=' ')
                csv_writer.writerows(img_right_patch_ann_bbs_yolo_format)

            if SAVE:
                pil_img = Image.fromarray(img_right_patch).convert('RGB')
                image_draw = ImageDraw.Draw(pil_img)
                if not os.path.exists("./output/gt/bbs_patches"):
                    os.makedirs("./output/gt/bbs_patches")
                for bb in img_right_patch_ann_bbs_yolo_format:
                    x_center = bb[0] * PATCH_WIDTH
                    y_center = bb[1] * PATCH_HEIGHT
                    bb_width = bb[2] * PATCH_WIDTH
                    bb_height = bb[3] * PATCH_HEIGHT
                    x_min = x_center - (bb_width / 2.0)
                    x_max = x_min + bb_width
                    y_min = y_center - (bb_height / 2.0)
                    y_max = y_min + bb_height
                    image_draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
                pil_img.save(os.path.join("./output/gt/bbs_patches", img_patch_name))
                if not os.path.exists("./output/gt/dmaps"):
                    os.makedirs("./output/gt/dmaps")
                Image.fromarray(normalize(dmap_right_patch).astype('uint8')). \
                    save(os.path.join("./output/gt/dmaps", img_patch_name.rsplit(".", 1)[0] + ".png"))

    df = pd.DataFrame({'imageName': names, 'X': ordinate, 'Y': abscissa})
    df.to_csv(DST_ANN_CSV_FILE, index=False)

    print("Exiting...")
