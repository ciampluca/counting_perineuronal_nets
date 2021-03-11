import os
from tifffile import imread
import numpy as np
from PIL import ImageDraw, Image

import torch
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor

from utils import transforms_bbs as custom_T


class PerineuralNetsBBoxDataset(VisionDataset):

    def __init__(self, data_root, transforms=None, list_frames=None):
        super().__init__(data_root, transforms)

        self.resize_factor = 32
        self.path_imgs = os.path.join(data_root, 'fullFrames')
        self.path_targets = os.path.join(data_root, 'annotation', 'bbs')

        self.imgs = sorted([file for file in os.listdir(self.path_imgs) if file.endswith(".tif")])
        if list_frames is not None:
            self.imgs = sorted([file for file in self.imgs if file.split("_", 1)[0] in list_frames])
        self.targets = sorted([file.rsplit(".", 1)[0] + ".txt" for file in self.imgs])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # Loading image
        img = imread(os.path.join(self.path_imgs, self.imgs[index]))
        img_h, img_w = img.shape[:2]
        # Eventually converting to 3 channels
        img = np.stack((img,) * 3, axis=-1)

        # Loading target
        # load target; from normalized x,y,w,h (x,y of the center) to denormalized xyxy (top left bottom right)
        target_path = os.path.join(self.path_targets, self.targets[index])
        bounding_boxes, bounding_boxes_areas = [], []
        num_bbs = 0
        with open(target_path, 'r') as bounding_box_file:
            for line in bounding_box_file:
                x_center = float(line.split()[0]) * img_w
                y_center = float(line.split()[1]) * img_h
                bb_width = float(line.split()[2]) * img_w
                bb_height = float(line.split()[3]) * img_h
                x_min = x_center - (bb_width / 2.0)
                x_max = x_min + bb_width
                y_min = y_center - (bb_height / 2.0)
                y_max = y_min + bb_height
                bounding_boxes.append([x_min, y_min, x_max, y_max])
                area = (y_max - y_min) * (x_max - x_min)
                bounding_boxes_areas.append(area)
                num_bbs += 1

        # Converting everything related to the target into a torch.Tensor
        bounding_boxes = torch.as_tensor(bounding_boxes, dtype=torch.float32)
        bounding_boxes_areas = torch.as_tensor(bounding_boxes_areas, dtype=torch.float32)
        labels = torch.ones((num_bbs,), dtype=torch.int64)  # there is only one class
        image_id = torch.tensor([index])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_bbs,), dtype=torch.int64)

        # Building target
        target = {}
        target["boxes"] = bounding_boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = bounding_boxes_areas
        target["iscrowd"] = iscrowd

        # Applying transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def standard_collate_fn(self, batch):
        return list(zip(*batch))


# Testing code
if __name__ == "__main__":

    def crop(img, height, width):
        imgwidth, imgheight = img.size
        for i in range(imgheight // height):
            for j in range(imgwidth // width):
                box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
                yield img.crop(box)

    NUM_WORKERS = 0
    BATCH_SIZE = 1
    DEVICE = "cpu"
    SHUFFLE = True
    CROP_WIDTH = 640
    CROP_HEIGHT = 640
    data_root = "/home/luca/luca-cnr/mnt/Dati_SSD_2/datasets/perineural_nets"
    train_frames = ['014', '015', '016', '017', '020', '021', '022', '023', '027', '028', '034', '035', '041', '042',
                    '043', '044', '049', '050', '051', '052']
    val_frames = ['019', '026', '036', '048', '053']

    transforms = custom_T.Compose([
        custom_T.RandomHorizontalFlip(),
        custom_T.RandomCrop(width=CROP_WIDTH, height=CROP_HEIGHT),
        custom_T.ToTensor(),
    ])

    val_transforms = custom_T.Compose([
        custom_T.PadToResizeFactor(resize_factor=CROP_WIDTH),
        custom_T.ToTensor(),
    ])

    dataset = PerineuralNetsBBoxDataset(data_root=data_root, transforms=transforms)

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        collate_fn=dataset.standard_collate_fn,
    )

    val_dataset = PerineuralNetsBBoxDataset(data_root=data_root, transforms=val_transforms, list_frames=val_frames)

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
            img_name = dataset.imgs[img_id]

            pil_image = to_pil_image(image.cpu())
            pil_image.save("./output/dataloading/{}.png".format(img_name.rsplit(".", 1)[0]))

            draw = ImageDraw.Draw(pil_image)
            for bb in target['boxes']:
                draw.rectangle([bb[0].item(), bb[1].item(), bb[2].item(), bb[3].item()])
            pil_image.save("./output/dataloading/{}_withBBs.png".format(img_name.rsplit(".", 1)[0]))

    # Validation
    for images, targets in val_data_loader:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        for image, target in zip(images, targets):
            img_id = target['image_id'].item()
            img_name = val_dataset.imgs[img_id]
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
                    img_patch = image[:, i:i + CROP_HEIGHT, j:j + CROP_WIDTH]
                    img_patch_bbs = []
                    for bb in bboxes:
                        bb_w, bb_h = bb[2] - bb[0], bb[3] - bb[1]
                        x_c, y_c = bb[0] + bb_w / 2, bb[1] + bb_h / 2
                        if j < x_c < j + CROP_WIDTH and i < y_c < i + CROP_HEIGHT:
                            x_tl, y_tl, x_br, y_br = bb[0] - j, bb[1] - i, bb[2] - j, bb[3] - i
                            if x_tl <= 0:
                                x_tl = 0
                            if y_tl <= 0:
                                y_tl = 0
                            if x_tl >= CROP_WIDTH:
                                x_tl = CROP_WIDTH
                            if y_tl >= CROP_HEIGHT:
                                y_tl = CROP_HEIGHT
                            img_patch_bbs.append([x_tl, y_tl, x_br, y_br])
                    output_patches.append(img_patch)

                    pil_image = to_pil_image(img_patch)
                    draw = ImageDraw.Draw(pil_image)
                    for bb in img_patch_bbs:
                        draw.rectangle([bb[0], bb[1], bb[2], bb[3]])
                    img_patch_bbs = torch.as_tensor(img_patch_bbs, dtype=torch.float32)
                    output_patches_withBBs.append(to_tensor(pil_image))
                    pil_image.save(
                        "./output/dataloading/{}_{}_withBBs.png".format(img_name.rsplit(".", 1)[0], counter_patches))
                    counter_patches += 1

            # Reconstructing image and image with BBs
            output_patches = torch.stack(output_patches)
            rec_image = output_patches.view(int(img_h / CROP_HEIGHT), int(img_w / CROP_WIDTH), *output_patches.size()[-3:])
            permuted_rec_image = rec_image.permute(2, 0, 3, 1, 4).contiguous()
            permuted_rec_image = permuted_rec_image.view(permuted_rec_image.shape[0], permuted_rec_image.shape[1] * permuted_rec_image.shape[2],
                                       permuted_rec_image.shape[3] * permuted_rec_image.shape[4])
            to_pil_image(permuted_rec_image).save("./output/dataloading/reconstructed_{}.png".format(img_name.rsplit(".", 1)[0]))

            output_patches_withBBs = torch.stack(output_patches_withBBs)
            rec_image_withBBs = output_patches_withBBs.view(int(img_h / CROP_HEIGHT), int(img_w / CROP_WIDTH),
                                            *output_patches_withBBs.size()[-3:])
            permuted_rec_image_withBBs = rec_image_withBBs.permute(2, 0, 3, 1, 4).contiguous()
            permuted_rec_image_withBBs = permuted_rec_image_withBBs.view(permuted_rec_image_withBBs.shape[0],
                                                         permuted_rec_image_withBBs.shape[1] * permuted_rec_image_withBBs.shape[2],
                                                         permuted_rec_image_withBBs.shape[3] * permuted_rec_image_withBBs.shape[4])
            to_pil_image(permuted_rec_image_withBBs).save(
                "./output/dataloading/reconstructed_{}_withBBS.png".format(img_name.rsplit(".", 1)[0]))



