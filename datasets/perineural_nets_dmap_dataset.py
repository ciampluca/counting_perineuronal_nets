import os
from tifffile import imread
import numpy as np
from PIL import Image

import torch
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
import torch.nn.functional as nnf
from torchvision.transforms.functional import to_pil_image

from utils import transforms_dmaps as custom_T
from utils.misc import normalize


class PerineuralNetsDmapDataset(VisionDataset):

    def __init__(self, data_root, transforms=None, list_frames=None):
        super().__init__(data_root, transforms)

        self.path_imgs = os.path.join(data_root, 'fullFrames')
        self.path_dmaps = os.path.join(data_root, 'annotation', 'dmaps')

        self.imgs = sorted([file for file in os.listdir(self.path_imgs) if file.endswith(".tif")])
        if list_frames is not None:
            self.imgs = sorted([file for file in self.imgs if file.split("_", 1)[0] in list_frames])

        self.dmaps = sorted([file.rsplit(".", 1)[0] + ".npy" for file in self.imgs])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # Loading image
        img = imread(os.path.join(self.path_imgs, self.imgs[index]))
        # Eventually converting to 3 channels
        img = np.stack((img,) * 3, axis=-1)

        # Loading dmap
        dmap = np.load(os.path.join(self.path_dmaps, self.dmaps[index])).astype(np.float32)

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
            pad_w = max_w - dmap.shape[1]
            pad_h = max_h - dmap.shape[0]
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
    train_frames = ['014', '015', '016', '017', '020', '021', '022', '023', '027', '028', '034', '035', '041', '042',
                   '043', '044', '049', '050', '051', '052']
    val_frames = ['019', '026', '036', '048', '053']
    data_root = "/home/luca/luca-cnr/mnt/Dati_SSD_2/datasets/perineural_nets"
    CROP_WIDTH = 2016
    CROP_HEIGHT = 2016

    assert CROP_WIDTH % 32 == 0 and CROP_HEIGHT % 32 == 0, "In validation mode, crop dim must be multiple of 32"

    train_transforms = custom_T.Compose([
            custom_T.RandomHorizontalFlip(),
            custom_T.RandomCrop(width=CROP_WIDTH, height=CROP_HEIGHT),
            custom_T.PadToResizeFactor(),
            custom_T.ToTensor(),
    ])

    val_transforms = custom_T.Compose([
        custom_T.PadToResizeFactor(resize_factor=CROP_WIDTH),
        custom_T.ToTensor(),
    ])

    dataset = PerineuralNetsDmapDataset(data_root=data_root, transforms=train_transforms, list_frames=train_frames)

    data_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=dataset.custom_collate_fn,
    )

    val_dataset = PerineuralNetsDmapDataset(data_root=data_root, transforms=val_transforms, list_frames=val_frames)

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=val_dataset.custom_collate_fn,
    )

    # Validation
    for images, targets in val_data_loader:
        images = images.to(DEVICE)
        gt_dmaps = targets['dmap'].unsqueeze(1).to(DEVICE)
        img_name = val_dataset.imgs[targets['img_id']]
        print(img_name)

        to_pil_image(images[0].cpu()).save("./output/dataloading/{}.png".format(img_name.rsplit(".", 1)[0]))
        Image.fromarray(normalize(gt_dmaps[0][0].cpu().numpy()).astype('uint8')). \
            save(os.path.join("./output/dataloading", img_name.rsplit(".", 1)[0] + "_dmap.png"))

        # Image is divided in patches
        output_patches = []
        patches = images.data.unfold(1, 3, 3).unfold(2, CROP_WIDTH, CROP_HEIGHT).unfold(3, CROP_WIDTH, CROP_HEIGHT)
        counter_patches = 1
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                for r in range(patches.shape[2]):
                    for c in range(patches.shape[3]):
                        img_patch = patches[i, j, r, c, ...]
                        output_patches.append(img_patch)
                        to_pil_image(img_patch).save("./output/dataloading/{}_{}.png".format(img_name.rsplit(".", 1)[0], counter_patches))
                        counter_patches += 1

        # Reconstructing image
        output_patches = torch.stack(output_patches)
        rec_image = output_patches.view(patches.shape[2], patches.shape[3], *patches.size()[-3:])
        rec_image = rec_image.permute(2, 0, 3, 1, 4).contiguous()
        rec_image = rec_image.view(rec_image.shape[0], rec_image.shape[1]*rec_image.shape[2], rec_image.shape[3]*rec_image.shape[4])

        to_pil_image(rec_image).save("./output/dataloading/reconstructed_{}.png".format(img_name.rsplit(".", 1)[0]))

        # The same for the target...
        output_patches = []
        patches = gt_dmaps.data.unfold(1, 1, 1).unfold(2, CROP_WIDTH, CROP_HEIGHT).unfold(3, CROP_WIDTH, CROP_HEIGHT)
        counter_patches = 1
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                for r in range(patches.shape[2]):
                    for c in range(patches.shape[3]):
                        target_patch = patches[i, j, r, c, ...]
                        output_patches.append(target_patch)
                        Image.fromarray(normalize(target_patch[0].cpu().numpy()).astype('uint8')).save(
                            "./output/dataloading/{}_{}_dmap.png".format(img_name.rsplit(".", 1)[0], counter_patches)
                        )
                        counter_patches += 1

        output_patches = torch.stack(output_patches)
        rec_target = output_patches.view(patches.shape[2], patches.shape[3], *patches.size()[-3:])
        rec_target = rec_target.permute(2, 0, 3, 1, 4).contiguous()
        rec_target = rec_target.view(rec_target.shape[0], rec_target.shape[1] * rec_target.shape[2],
                                   rec_target.shape[3] * rec_target.shape[4])

        Image.fromarray(normalize(rec_target[0].cpu().numpy()).astype('uint8')). \
            save(os.path.join("./output/dataloading", "reconstructed_" + img_name.rsplit(".", 1)[0] + "_dmap.png"))


