import os
from tifffile import imread
import numpy as np
from PIL import ImageDraw, Image
import copy
import tqdm
import albumentations.augmentations.bbox_utils as albumentations_utils

import torch
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor

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
        labels = torch.ones((len(bounding_boxes),), dtype=torch.int64)  # there is only one class
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(bounding_boxes),), dtype=torch.int64)

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
    data_root = "/home/luca/luca-cnr/mnt/Dati_SSD_2/datasets/perineural_nets"
    train_frames = ['014', '015', '017', '019', '020', '021', '023', '026', '027', '028', '035', '036', '041', '042', '044', '048', '049', '050', '052', '053']
    val_frames = ['016', '022', '034', '043', '051']
    all_frames = ['014', '015', '016', '017', '019', '020', '021', '022', '023', '026', '027', '028', '034', '035', '036', '041', '042', '043', '044', '048', '049', '050', '051', '052', '053']
    if SPECULAR_SPLIT:
        train_frames = val_frames = all_frames

    transforms = custom_T.Compose([
        custom_T.RandomHorizontalFlip(),
        custom_T.RandomCrop(width=CROP_WIDTH, height=CROP_HEIGHT, min_visibility=0.5),
        custom_T.ToTensor(),
    ])

    val_transforms = custom_T.Compose([
        custom_T.PadToResizeFactor(resize_factor=CROP_WIDTH),
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
    for images, targets in data_loader:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        for image, target in zip(images, targets):
            img_id = target['image_id'].item()
            img_name = dataset.image_files[img_id]

            pil_image = to_pil_image(image.cpu())
            draw = ImageDraw.Draw(pil_image)
            for bb in target['boxes']:
                draw.rectangle([bb[0].item(), bb[1].item(), bb[2].item(), bb[3].item()], outline='red', width=3)
            pil_image.save("./output/dataloading/{}_withBBs.png".format(img_name.rsplit(".", 1)[0]))

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
                draw.rectangle([bb[0], bb[1], bb[2], bb[3]])
            pil_image.save("./output/dataloading/{}_withBBs.png".format(img_name.rsplit(".", 1)[0]))

            # Image is divided in patches
            img_w, img_h = image.shape[2], image.shape[1]
            output_patches, output_patches_withBBs = [], []
            counter_patches = 1
            for i in range(0, img_h, CROP_HEIGHT):
                for j in range(0, img_w, CROP_WIDTH):
                    image_patch, target_patch = custom_T.CropToFixedSize()(
                        image,
                        x_min=j,
                        y_min=i,
                        x_max=j + CROP_WIDTH,
                        y_max=i + CROP_HEIGHT,
                        min_visibility=0.5,
                        target=copy.deepcopy(target)
                    )
                    output_patches.append(image_patch)

                    pil_image = to_pil_image(image_patch)
                    draw = ImageDraw.Draw(pil_image)
                    for bb in target_patch['boxes']:
                        draw.rectangle([bb[0].item(), bb[1].item(), bb[2].item(), bb[3].item()], outline='red', width=3)
                    output_patches_withBBs.append(to_tensor(pil_image))
                    pil_image.save(
                        "./output/dataloading/{}_{}_withBBs.png".format(img_name.rsplit(".", 1)[0], counter_patches))
                    counter_patches += 1

            # Reconstructing image with BBs
            output_patches_withBBs = torch.stack(output_patches_withBBs)
            rec_image_withBBs = output_patches_withBBs.view(int(img_h / CROP_HEIGHT), int(img_w / CROP_WIDTH),
                                            *output_patches_withBBs.size()[-3:])
            permuted_rec_image_withBBs = rec_image_withBBs.permute(2, 0, 3, 1, 4).contiguous()
            permuted_rec_image_withBBs = permuted_rec_image_withBBs.view(permuted_rec_image_withBBs.shape[0],
                                                         permuted_rec_image_withBBs.shape[1] * permuted_rec_image_withBBs.shape[2],
                                                         permuted_rec_image_withBBs.shape[3] * permuted_rec_image_withBBs.shape[4])
            to_pil_image(permuted_rec_image_withBBs).save(
                "./output/dataloading/reconstructed_{}_withBBS.png".format(img_name.rsplit(".", 1)[0]))



