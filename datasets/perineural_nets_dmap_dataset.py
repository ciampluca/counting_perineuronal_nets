import os
from tifffile import imread
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tqdm
import math
import copy

import torch
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
import torch.nn.functional as nnf
from torchvision.transforms.functional import to_pil_image

from utils import transforms_dmaps as custom_T
from utils.misc import normalize


class PerineuralNetsDmapDataset(VisionDataset):

    def __init__(self, data_root, transforms=None, list_frames=None, with_patches=True, load_in_memory=False,
                 percentage=None, dataset_name=None, specular_split=True):
        super().__init__(data_root, transforms)

        self.resize_factor = 32
        self.load_in_memory = load_in_memory
        if dataset_name:
            self.dataset_name = dataset_name

        if with_patches and not specular_split:
            self.path_imgs = os.path.join(data_root, 'random_patches')
            self.path_targets = os.path.join(data_root, 'annotation', 'random_patches_dmaps')
        elif with_patches and specular_split:
            self.path_imgs = os.path.join(data_root, 'specular_patches')
            self.path_targets = os.path.join(data_root, 'annotation', 'specular_patches_dmaps')
        elif not with_patches and not specular_split:
            self.path_imgs = os.path.join(data_root, 'fullFrames')
            self.path_targets = os.path.join(data_root, 'annotation', 'dmaps')
        elif not with_patches and specular_split:
            self.path_imgs = os.path.join(data_root, 'specular_fullFrames')
            self.path_targets = os.path.join(data_root, 'annotation', 'specular_dmaps')

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
            self.images, self.dmaps = [], []
            for img_f in tqdm.tqdm(self.image_files):
                img, dmap = self._load_sample(img_f)
                self.images.append(img)
                self.dmaps.append(dmap)

    def _load_sample(self, img_f):
        # Loading image
        img = imread(os.path.join(self.path_imgs, img_f))
        img = np.stack((img,) * 3, axis=-1)

        # Loading dmap
        dmap = np.load(os.path.join(self.path_targets, img_f.rsplit(".", 1)[0] + ".npy")).astype(np.float32)

        return img, dmap

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        if self.load_in_memory:
            img, dmap = self.images[index], self.dmaps[index]
        else:
            img_f = self.image_files[index]
            img, dmap = self._load_sample(img_f)

        # Applying transforms
        if self.transforms is not None:
            img, dmap = self.transforms(img, dmap)

        # Building target
        target = {}
        target['dmap'] = dmap
        target['img_id'] = torch.as_tensor(index, dtype=torch.int32)

        return img, target

    def custom_collate_fn(self, batch):
        # Padding images to the max size of the image in the batch.
        imgs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        shapes = [list(img.shape) for img in imgs]

        maxes = shapes[0]
        for sublist in shapes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)

        max_h = maxes[1]
        max_w = maxes[2]

        padded_images = []
        for img in imgs:
            pad_w = max_w - img.shape[2]
            pad_h = max_h - img.shape[1]
            p2d = (int(pad_w / 2), pad_w - int(pad_w / 2), int(pad_h / 2), pad_h - int(pad_h / 2))
            padded_images.append(nnf.pad(img, p2d, "constant", 0))
        collated_images = torch.stack(padded_images, dim=0)

        padded_dmaps, img_ids = [], []
        for target in targets:
            dmap = target['dmap']
            pad_w = max_w - dmap.shape[2]
            pad_h = max_h - dmap.shape[1]
            p2d = (int(pad_w / 2), pad_w - int(pad_w / 2), int(pad_h / 2), pad_h - int(pad_h / 2))
            padded_dmaps.append(nnf.pad(dmap, p2d, "constant", 0))
            img_ids.append(target['img_id'])

        collated_targets = dict()
        collated_targets['dmap'] = torch.stack(padded_dmaps, dim=0)
        collated_targets['img_id'] = torch.stack(img_ids)

        return [collated_images, collated_targets]


# Testing code
if __name__ == "__main__":
    NUM_WORKERS = 0
    BATCH_SIZE = 1
    DEVICE = "cpu"
    SPECULAR_SPLIT = True
    train_frames = ['014', '015', '017', '019', '020', '021', '023', '026', '027', '028', '035', '036', '041', '042', '044', '048', '049', '050', '052', '053']
    val_frames = ['016', '022', '034', '043', '051']
    all_frames = ['014', '015', '016', '017', '019', '020', '021', '022', '023', '026', '027', '028', '034', '035', '036', '041', '042', '043', '044', '048', '049', '050', '051', '052', '053']
    if SPECULAR_SPLIT:
        train_frames = val_frames = all_frames
    data_root = "/mnt/Dati_SSD_2/datasets/perineural_nets"
    CROP_WIDTH = 640
    CROP_HEIGHT = 640
    STRIDE_W, STRIDE_H = CROP_WIDTH, CROP_HEIGHT
    OVERLAPPING_PATCHES = True
    if OVERLAPPING_PATCHES:
        STRIDE_W, STRIDE_H = CROP_WIDTH - 120, CROP_HEIGHT - 120

    assert CROP_WIDTH % 32 == 0 and CROP_HEIGHT % 32 == 0, "In validation mode, crop dim must be multiple of 32"

    train_transforms = custom_T.Compose([
            custom_T.RandomHorizontalFlip(),
            custom_T.RandomCrop(width=CROP_WIDTH, height=CROP_HEIGHT),
            custom_T.PadToResizeFactor(),
            custom_T.ToTensor(),
    ])

    val_transforms = custom_T.Compose([
        # custom_T.PadToResizeFactor(resize_factor=CROP_WIDTH),
        custom_T.ToTensor(),
    ])

    dataset = PerineuralNetsDmapDataset(
        data_root=data_root,
        transforms=train_transforms,
        list_frames=train_frames,
        with_patches=True,
        load_in_memory=False,
        specular_split=SPECULAR_SPLIT,
    )

    data_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=dataset.custom_collate_fn,
    )

    val_dataset = PerineuralNetsDmapDataset(
        data_root=data_root,
        transforms=val_transforms,
        list_frames=val_frames,
        with_patches=False,
        load_in_memory=False,
        specular_split=SPECULAR_SPLIT,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=val_dataset.custom_collate_fn,
    )

    # Training
    # for images, targets in data_loader:
    #     images = list(image.to(DEVICE) for image in images)
    #     gt_dmaps = list(dmap.to(DEVICE) for dmap in targets['dmap'])
    #     img_names = list(dataset.image_files[img_id] for img_id in targets['img_id'])
    #
    #     for img, dmap, img_name in zip(images, gt_dmaps, img_names):
    #         pil_image = to_pil_image(img.cpu())
    #         pil_image.save("./output/dataloading/{}.png".format(img_name.rsplit(".", 1)[0]))
    #         pil_dmap = Image.fromarray(normalize(dmap.squeeze(dim=0).cpu().numpy()).astype('uint8'))
    #         pil_dmap.save("./output/dataloading/{}_dmap.png".format(img_name.rsplit(".", 1)[0]))

    # Validation
    for images, targets in val_data_loader:
        images = images.to(DEVICE)
        gt_dmaps = targets['dmap'].to(DEVICE)
        img_name = val_dataset.image_files[targets['img_id']]
        print(img_name)
        # Batch size in val mode is always 1
        image = images.squeeze(dim=0)
        gt_dmap = gt_dmaps.squeeze(dim=0)
        to_pil_image(image).save("./output/dataloading/original_{}.png".format(img_name.rsplit(".", 1)[0]))
        num_nets = torch.sum(gt_dmap)
        pil_den_map = Image.fromarray(normalize(copy.deepcopy(gt_dmap).squeeze(dim=0).cpu().numpy()).astype('uint8'))
        draw = ImageDraw.Draw(pil_den_map)
        # Add text to image
        text = "Num of Nets: {}".format(num_nets)
        font_path = "./font/LEMONMILK-RegularItalic.otf"
        font = ImageFont.truetype(font_path, 100)
        draw.text((75, 75), text=text, font=font, fill=191)
        pil_den_map.save(
            "./output/dataloading/original_{}_dmap.png".format(img_name.rsplit(".", 1)[0])
        )

        # Image and dmap are divided in patches
        img_w, img_h = image.shape[2], image.shape[1]
        num_h_patches, num_v_patches = math.ceil(img_w / STRIDE_W), math.ceil(img_h / STRIDE_H)
        img_w_padded = (num_h_patches - math.floor(img_w / STRIDE_W)) * (STRIDE_W * num_h_patches + (CROP_WIDTH-STRIDE_W))
        img_h_padded = (num_v_patches - math.floor(img_h / STRIDE_H)) * (STRIDE_H * num_v_patches + (CROP_HEIGHT-STRIDE_H))
        padded_image, padded_gt_dmap = custom_T.PadToSize()(
            image=image,
            min_width=img_w_padded,
            min_height=img_h_padded,
            dmap=gt_dmap
        )

        h_pad_top = int((img_h_padded - img_h) / 2.0)
        h_pad_bottom = img_h_padded - img_h - h_pad_top
        w_pad_left = int((img_w_padded - img_w) / 2.0)
        w_pad_right = img_w_padded - img_w - w_pad_left

        normalization_map = torch.zeros_like(padded_image)
        reconstructed_image = torch.zeros_like(padded_image)
        reconstructed_dmap = torch.zeros_like(padded_gt_dmap)

        for i in range(0, img_h, STRIDE_H):
            for j in range(0, img_w, STRIDE_W):
                image_patch, dmap_patch = custom_T.CropToFixedSize()(
                    padded_image,
                    x_min=j,
                    y_min=i,
                    x_max=j + CROP_WIDTH,
                    y_max=i + CROP_HEIGHT,
                    dmap=copy.deepcopy(padded_gt_dmap)
                )

                reconstructed_image[:, i:i + CROP_HEIGHT, j:j + CROP_WIDTH] += image_patch
                reconstructed_dmap[:, i:i + CROP_HEIGHT, j:j + CROP_WIDTH] += dmap_patch
                normalization_map[:, i:i + CROP_HEIGHT, j:j + CROP_WIDTH] += 1.0

        reconstructed_image /= normalization_map
        reconstructed_dmap /= normalization_map[0].unsqueeze(dim=0)

        reconstructed_image = reconstructed_image[:, h_pad_top:img_h_padded-h_pad_bottom, w_pad_left:img_w_padded-w_pad_right]
        reconstructed_dmap = reconstructed_dmap[:, h_pad_top:img_h_padded-h_pad_bottom, w_pad_left:img_w_padded-w_pad_right]

        to_pil_image(reconstructed_image).save(
            "./output/dataloading/reconstructed_{}.png".format(img_name.rsplit(".", 1)[0]))
        num_nets = torch.sum(reconstructed_dmap)
        pil_reconstructed_dmap = Image.fromarray(normalize(reconstructed_dmap.squeeze(dim=0).cpu().numpy()).astype('uint8'))
        draw = ImageDraw.Draw(pil_reconstructed_dmap)
        # Add text to image
        text = "Num of Nets: {}".format(num_nets)
        font_path = "./font/LEMONMILK-RegularItalic.otf"
        font = ImageFont.truetype(font_path, 100)
        draw.text((75, 75), text=text, font=font, fill=191)
        pil_reconstructed_dmap.save(
            "./output/dataloading/reconstructed_{}_dmap.png".format(img_name.rsplit(".", 1)[0])
        )





