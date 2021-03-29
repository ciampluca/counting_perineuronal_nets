import os
from tifffile import imread
import numpy as np
from PIL import ImageDraw, Image, ImageFont
import copy
import tqdm
import albumentations.augmentations.bbox_utils as albumentations_utils
import math

import torch
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import boxes as box_ops

from utils import transforms_bbs as custom_T


class PerineuralNetsBBoxDataset(VisionDataset):

    def __init__(self, data_root, transforms=None, list_frames=None, with_patches=True, load_in_memory=False,
                 percentage=None, dataset_name=None, min_visibility=0.0, original_bb_dim=60, specular_split=True):
        super().__init__(data_root, transforms)

        self.resize_factor = 32
        self.load_in_memory = load_in_memory
        self.min_visibility = min_visibility
        self.original_bb_dim = original_bb_dim
        if dataset_name:
            self.dataset_name = dataset_name

        if with_patches and not specular_split:
            self.path_imgs = os.path.join(data_root, 'random_patches')
            self.path_targets = os.path.join(data_root, 'annotation', 'random_patches_bbs')
        elif with_patches and specular_split:
            self.path_imgs = os.path.join(data_root, 'specular_patches')
            self.path_targets = os.path.join(data_root, 'annotation', 'specular_patches_bbs')
        elif not with_patches and not specular_split:
            self.path_imgs = os.path.join(data_root, 'fullFrames')
            self.path_targets = os.path.join(data_root, 'annotation', 'bbs')
        elif not with_patches and specular_split:
            self.path_imgs = os.path.join(data_root, 'specular_fullFrames')
            self.path_targets = os.path.join(data_root, 'annotation', 'specular_bbs')

        self.image_files = sorted([file for file in os.listdir(self.path_imgs) if file.endswith(".tif")])

        if list_frames is not None:
            self.image_files = sorted([file for file in self.image_files if file.split("_", 1)[0] in list_frames])
            if specular_split and with_patches:
                left_frames, right_frames = list_frames[::2], list_frames[1::2]
                left_image_files = sorted([file for file in self.image_files
                                           if file.split("_", 1)[0] in left_frames and file.split("_")[4] == "left"])
                right_image_files = sorted([file for file in self.image_files
                                           if file.split("_", 1)[0] in right_frames and file.split("_")[4] == "right"])
                self.image_files = left_image_files + right_image_files
            elif specular_split and not with_patches:
                right_frames, left_frames = list_frames[::2], list_frames[1::2]
                left_image_files = sorted([file for file in self.image_files
                                           if file.split("_", 1)[0] in left_frames and file.split("_")[4].rsplit(".", 1)[0] == "left"])
                right_image_files = sorted([file for file in self.image_files
                                           if file.split("_", 1)[0] in right_frames and file.split("_")[4].rsplit(".", 1)[0] == "right"])
                self.image_files = left_image_files + right_image_files

        if percentage is not None:
            # only keep num images of the provided percentage
            num_images = int((len(self.image_files) / 100) * percentage)
            indices = torch.randperm(len(self.image_files)).tolist()
            indices = indices[-num_images:]
            self.image_files = [self.image_files[index] for index in indices]

        if load_in_memory:
            print("Loading dataset in memory!")
            # load all the data into memory
            self.images, self.targets = [], []
            for img_f in tqdm.tqdm(self.image_files):
                img, target = self._load_sample(img_f)
                self.images.append(img)
                self.targets.append(target)

    def _load_sample(self, img_f):
        # Loading image
        img = imread(os.path.join(self.path_imgs, img_f))
        img_h, img_w = img.shape[:2]
        img = np.stack((img,) * 3, axis=-1)

        # Loading target
        bounding_boxes_yolo_format = []
        with open(os.path.join(self.path_targets, img_f.rsplit(".", 1)[0] + ".txt"), 'r') as bounding_box_file:
            for line in bounding_box_file:
                x_center = float(line.split()[0])
                y_center = float(line.split()[1])
                bb_width = float(line.split()[2])
                bb_height = float(line.split()[3])
                bounding_boxes_yolo_format.append([x_center, y_center, bb_width, bb_height])

        bounding_boxes = self._convert_and_check(bounding_boxes_yolo_format, img_w, img_h)
        bounding_boxes = self._filter_bbs_by_visibility(bounding_boxes, self.min_visibility, self.original_bb_dim)
        bounding_boxes_areas = self._compute_bb_areas(bounding_boxes)

        # Converting everything related to the target into a torch.Tensor
        bounding_boxes = torch.as_tensor(bounding_boxes, dtype=torch.float32)
        bounding_boxes_areas = torch.as_tensor(bounding_boxes_areas, dtype=torch.float32)
        labels = torch.ones((len(bounding_boxes),), dtype=torch.int16)  # there is only one class
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(bounding_boxes),), dtype=torch.int16)

        # Building target
        target = {
            "boxes": bounding_boxes,
            "labels": labels,
            "area": bounding_boxes_areas,
            "iscrowd": iscrowd,
        }

        return img, target

    def _convert_and_check(self, bounding_boxes_yolo_format, img_w, img_h):
        # Converts from yolo format to xmin, ymin, xmax, ymax and checks validity

        # Converting to albumentations format and checking validity
        bounding_boxes_yolo_format = [tuple(bb) for bb in bounding_boxes_yolo_format]
        bounding_boxes_alb_format = albumentations_utils.convert_bboxes_to_albumentations(
            bboxes=bounding_boxes_yolo_format,
            source_format='yolo',
            rows=img_h,
            cols=img_w,
            check_validity=True,
        )

        # Converting to pascal_voc format and checking validity
        bounding_boxes = albumentations_utils.convert_bboxes_from_albumentations(
            bboxes=bounding_boxes_alb_format,
            target_format='pascal_voc',
            rows=img_h,
            cols=img_w,
            check_validity=True,
        )

        bounding_boxes = [list(elem) for elem in bounding_boxes]

        return bounding_boxes

    def _compute_bb_areas(self, bbs):
        bb_areas = []
        for bb in bbs:
            x_min, y_min, x_max, y_max = bb[:4]
            bb_areas.append((x_max - x_min) * (y_max - y_min))

        return bb_areas

    def _filter_bbs_by_visibility(self, bounding_boxes, min_visibility, original_bb_dim):
        if min_visibility == 0.0:
            return bounding_boxes

        filtered_bbs = []
        original_area = original_bb_dim*original_bb_dim
        for bb in bounding_boxes:
            x_min, y_min, x_max, y_max = bb[:4]
            area = (x_max - x_min) * (y_max - y_min)
            if (area / original_area) < min_visibility:
                continue
            filtered_bbs.append(bb)

        return filtered_bbs

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        if self.load_in_memory:
            img, target = self.images[index], self.targets[index]
        else:
            img_f = self.image_files[index]
            img, target = self._load_sample(img_f)

        # Adding img_id to target
        image_id = torch.tensor([index])
        target["image_id"] = image_id

        # Applying transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def standard_collate_fn(self, batch):
        return list(zip(*batch))


# Testing code
if __name__ == "__main__":

    NUM_WORKERS = 0
    BATCH_SIZE = 1
    DEVICE = "cpu"
    SHUFFLE = True
    CROP_WIDTH = 640
    CROP_HEIGHT = 640
    SPECULAR_SPLIT = True
    data_root = "/mnt/Dati_SSD_2/datasets/perineural_nets"
    train_frames = ['014', '015', '017', '019', '020', '021', '023', '026', '027', '028', '035', '036', '041', '042', '044', '048', '049', '050', '052', '053']
    val_frames = ['016', '022', '034', '043', '051']
    all_frames = ['014', '015', '016', '017', '019', '020', '021', '022', '023', '026', '027', '028', '034', '035', '036', '041', '042', '043', '044', '048', '049', '050', '051', '052', '053']
    if SPECULAR_SPLIT:
        train_frames = val_frames = all_frames
    STRIDE_W, STRIDE_H = CROP_WIDTH, CROP_HEIGHT
    OVERLAPPING_PATCHES = True
    if OVERLAPPING_PATCHES:
        STRIDE_W, STRIDE_H = CROP_WIDTH - 120, CROP_HEIGHT - 120

    transforms = custom_T.Compose([
        custom_T.RandomHorizontalFlip(),
        custom_T.RandomCrop(width=CROP_WIDTH, height=CROP_HEIGHT, min_visibility=0.5),
        custom_T.ToTensor(),
    ])

    val_transforms = custom_T.Compose([
        # custom_T.PadToResizeFactor(resize_factor=CROP_WIDTH),
        custom_T.ToTensor(),
    ])

    dataset = PerineuralNetsBBoxDataset(
        data_root=data_root,
        transforms=transforms,
        with_patches=True,
        load_in_memory=False,
        list_frames=train_frames,
        min_visibility=0.5,
        specular_split=SPECULAR_SPLIT,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        collate_fn=dataset.standard_collate_fn,
    )

    val_dataset = PerineuralNetsBBoxDataset(
        data_root=data_root,
        transforms=val_transforms,
        with_patches=False,
        load_in_memory=False,
        list_frames=val_frames,
        min_visibility=0.5,
        specular_split=SPECULAR_SPLIT,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=val_dataset.standard_collate_fn,
    )

    # Training
    # for images, targets in data_loader:
    #     images = list(image.to(DEVICE) for image in images)
    #     targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    #
    #     for image, target in zip(images, targets):
    #         img_id = target['image_id'].item()
    #         img_name = dataset.image_files[img_id]
    #
    #         pil_image = to_pil_image(image.cpu())
    #         draw = ImageDraw.Draw(pil_image)
    #         for bb in target['boxes']:
    #             draw.rectangle([bb[0].item(), bb[1].item(), bb[2].item(), bb[3].item()], outline='red', width=3)
    #         pil_image.save("./output/dataloading/{}_withBBs.png".format(img_name.rsplit(".", 1)[0]))

    # Validation
    for images, targets in val_data_loader:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        for image, target in zip(images, targets):
            img_id = target['image_id'].item()
            img_name = val_dataset.image_files[img_id]
            bboxes = target['boxes'].tolist()

            pil_image = to_pil_image(image.cpu())
            draw = ImageDraw.Draw(pil_image)
            for bb in bboxes:
                draw.rectangle([bb[0], bb[1], bb[2], bb[3]], outline='red', width=3)
            pil_image.save("./output/dataloading/original_{}_withBBs.png".format(img_name.rsplit(".", 1)[0]))

            # Image is divided in patches
            img_w, img_h = image.shape[2], image.shape[1]

            num_h_patches, num_v_patches = math.ceil(img_w / STRIDE_W), math.ceil(img_h / STRIDE_H)
            img_w_padded = (num_h_patches - math.floor(img_w / STRIDE_W)) * (
                        STRIDE_W * num_h_patches + (CROP_WIDTH - STRIDE_W))
            img_h_padded = (num_v_patches - math.floor(img_h / STRIDE_H)) * (
                        STRIDE_H * num_v_patches + (CROP_HEIGHT - STRIDE_H))
            padded_image, padded_target = custom_T.PadToSize()(
                image=image,
                min_width=img_w_padded,
                min_height=img_h_padded,
                target=copy.deepcopy(target)
            )

            normalization_map = torch.zeros_like(padded_image)
            overlapping_map = np.zeros((normalization_map.shape[1], normalization_map.shape[2]), dtype=np.uint8)
            reconstructed_image = torch.zeros_like(padded_image)

            for i in range(0, img_h, STRIDE_H):
                for j in range(0, img_w, STRIDE_W):
                    image_patch, target_patch = custom_T.CropToFixedSize()(
                        padded_image,
                        x_min=j,
                        y_min=i,
                        x_max=j + CROP_WIDTH,
                        y_max=i + CROP_HEIGHT,
                        min_visibility=0.5,
                        target=copy.deepcopy(padded_target)
                    )

                    reconstructed_image[:, i:i+CROP_HEIGHT, j:j+CROP_WIDTH] += image_patch
                    normalization_map[:, i:i+CROP_HEIGHT, j:j+CROP_WIDTH] += 1.0

            reconstructed_image /= normalization_map
            overlapping_map = np.where(normalization_map[0].cpu().numpy() != 1.0, 1, 0)

            gt_bbs = padded_target['boxes'].cpu().tolist()

            # Perform filtering of the bbs in the overlapped areas, for example nms
            # in this case we are using gt bbs, just for testing
            keep = []
            for i, gt_bb in enumerate(gt_bbs):
                bb_w, bb_h = gt_bb[2] - gt_bb[0], gt_bb[3] - gt_bb[1]
                x_c, y_c = int(gt_bb[0] + (bb_w/2)), int(gt_bb[1] + (bb_h/2))
                if overlapping_map[y_c, x_c] == 1:
                    keep.append(i)
            bbs_in_overlapped_areas = [gt_bbs[i] for i in keep]
            final_bbs = [bb for i, bb in enumerate(gt_bbs) if i not in keep]

            fake_scores = torch.ones(len(bbs_in_overlapped_areas), dtype=torch.float32)
            keep = box_ops.nms(
                torch.as_tensor(bbs_in_overlapped_areas, dtype=torch.float32),
                fake_scores,
                iou_threshold=0.001,
            )
            bbs_in_overlapped_areas = [bbs_in_overlapped_areas[i] for i in keep]
            final_bbs.extend(bbs_in_overlapped_areas)

            h_pad_top = int((img_h_padded - img_h) / 2.0)
            h_pad_bottom = img_h_padded - img_h - h_pad_top
            w_pad_left = int((img_w_padded - img_w) / 2.0)
            w_pad_right = img_w_padded - img_w - w_pad_left

            final_bbs = [
                [bb[0] - w_pad_left, bb[1] - h_pad_top, bb[2] - w_pad_left, bb[3] - h_pad_top] for
                bb in final_bbs]

            reconstructed_image = reconstructed_image[:, h_pad_top:img_h_padded - h_pad_bottom,
                                 w_pad_left:img_w_padded - w_pad_right]

            pil_reconstructed_image = to_pil_image(reconstructed_image)
            draw = ImageDraw.Draw(pil_reconstructed_image)
            for bb in final_bbs:
                draw.rectangle([bb[0], bb[1], bb[2], bb[3]], outline='red', width=3)
            # Add text to image
            text = "Num of Nets: {}".format(len(final_bbs))
            font_path = "./font/LEMONMILK-RegularItalic.otf"
            font = ImageFont.truetype(font_path, 100)
            draw.text((75, 75), text=text, font=font, fill=(0, 191, 255))
            pil_reconstructed_image.save(
                "./output/dataloading/reconstructed_{}_witBBs.png".format(img_name.rsplit(".", 1)[0]))





