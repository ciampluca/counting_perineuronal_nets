import os
from tqdm import tqdm
from skimage.io import imread
import numpy as np
import scipy.stats
from skimage.color import rgb2gray
from copy import deepcopy
import pandas as pd

import torch
from torch.utils.data import Dataset


class VGGCellsDataset(Dataset):

    # params for groundtruth bbs generation
    DEFAULT_GT_PARAMS = {
        'side': 12,         # side (in px) of the bounding box localizing a cell
        'win_size': 32,
        'scale': 1,
        'stride': 1,
        'cov': 1
    }

    def __init__(self, root, transforms=None, in_memory=False, image_names=None, gt_params={}, ann_type="det", **kwargs):

        assert ann_type in ['det', 'den', 'segm', None], "Not implemented annotation type"
        if image_names is None or not image_names:
            print(f"You have to pass a list of image names")
            exit(1)

        self.root = root
        self.transforms = transforms
        self.in_memory = in_memory
        self.image_names = image_names
        self.ann_type = ann_type

        # groundtruth parameters
        self.gt_params = deepcopy(self.DEFAULT_GT_PARAMS)
        self.gt_params.update(gt_params)

        # create pandas dataframe containing dot annotations (to be compliant with other implementation)
        self.annot = self._create_pd_dot_annot()

        if in_memory:
            print("Loading dataset in memory!")
            # load all the data into memory
            self.samples = []
            for img_n in tqdm(self.image_names):
                self.samples.append(self._load_sample(img_n))

    def __getitem__(self, index):
        if self.in_memory:
            sample = self.samples[index]
        else:
            img_n = self.image_names[index]
            sample = self._load_sample(img_n)

        if self.transforms is not None:
            sample = (self.transforms(sample[0]),) + sample[1:]

        return sample

    def custom_collate_fn(self, batch):
        return list(zip(*batch))

    def _load_sample(self, img_n):
        # Loading image
        img = rgb2gray(imread(os.path.join(self.root, img_n))).astype(np.float32)
        label = imread(os.path.join(self.root, img_n.replace("cell", "dots")))
        label = label[:, :, 0] / 255
        image_hw = img.shape

        if self.ann_type == "den":
            markers = self._get_markers(label, image_hw)
            img_pad = np.pad(img, self.gt_params['win_size'] // 2, "constant", constant_values=0)
            dmap = self._get_density(img_pad.shape[0:2], markers)
            # stack input and target
            input_and_target = np.stack((img_pad, dmap), axis=-1)
            datum = input_and_target
        elif self.ann_type == "det":
            points = np.nonzero(label)
            bbs = self._build_detection_target(img, points)
            # put in a unique tuple the patch and the target
            img = np.expand_dims(img, axis=-1)  # add channels dimension
            input_and_target = (img, bbs)
            datum = input_and_target
        elif self.ann_type == "segm":
            points = np.nonzero(label)
            

        else:   # no targets
            datum = np.expand_dims(img, axis=-1)  # add channels dimension

        # These variables are defined in order to be compliant with the perineural nets dataset.
        # To be implemented if one wants to use patches also with this dataset
        start_yx = (0, 0)
        patch_hw = image_hw

        return datum, patch_hw, start_yx, image_hw, img_n

    def _get_markers(self, label, shape):
        bin_size = [self.gt_params['scale'], self.gt_params['scale']]
        markers = np.zeros(shape)
        for i in range(bin_size[0]):
            for j in range(bin_size[1]):
                markers = np.maximum(label[i::bin_size[0], j::bin_size[1]], markers)

        assert np.allclose(label.sum(), markers.sum(), 1)

        markers = np.pad(markers, self.gt_params['win_size'], "constant", constant_values=-1)

        return markers

    def _get_density(self, dim, markers):
        height, width = ((dim[0]) // self.gt_params['stride']), ((dim[1]) // self.gt_params['stride'])
        markers = markers[0:0 + height, 0:0 + width]
        gaus_img = np.zeros((height, width))
        for k in range(height):
            for l in range(width):
                if markers[k, l] > 0.5:
                    gaus_img += self._gen_gaus_img(len(markers), k - self.gt_params['win_size'] / 2,
                                                   l - self.gt_params['win_size'] / 2, self.gt_params['win_size'])

        return gaus_img

    def _gen_gaus_img(self, win_size, mx, my, cov=1):
        x, y = np.mgrid[0:win_size, 0:win_size]
        pos = np.dstack((x, y))
        mean = [mx, my]
        cov = [[cov, 0], [0, cov]]
        rv = scipy.stats.multivariate_normal(mean, cov).pdf(pos)

        return rv / rv.sum()

    def _build_detection_target(self, img, locations):
        """ This builds the detection target
        """
        side = self.gt_params['side']
        half_side = side / 2

        shape = img.shape

        if len(locations) == 0:  # empty patch
            bbs = np.array([[]], dtype=np.float32)
        else:
            # bb format: [y1, x1, y2, x2]
            bbs = np.empty((len(locations[0]), 4), dtype=np.float32)
            for i, (y, x) in enumerate(zip(locations[0], locations[1])):
                center = (y, x)
                bbs[i] = [center[0]-half_side, center[1]-half_side, center[0]+half_side, center[1]+half_side]
                np.clip(bbs[i], [0, 0, 0, 0], [shape[0], shape[1], shape[0], shape[1]], out=bbs[i])

        return bbs

    def __len__(self):
        # total number of imgs
        return len(self.image_names)

    def build_coco_compliant_batch(self, image_and_target_batch):
        targets, imgs = [], []

        for b in image_and_target_batch:
            imgs.append(b[0])

            if b[1].size != 0:
                target = {
                    'boxes': torch.as_tensor([[bb[1], bb[0], bb[3], bb[2]] for bb in b[1]], dtype=torch.float32),
                    'labels': torch.ones((len(b[1]),), dtype=torch.int64),  # there is only one class
                    'iscrowd': torch.zeros((len(b[1]),), dtype=torch.int64),     # suppose all instances are not crowd
                }
            else:
                target = {
                    'boxes': torch.as_tensor([[]], dtype=torch.float32),
                    'labels': torch.as_tensor([[]], dtype=torch.int64),
                    'iscrowd': torch.as_tensor([[]], dtype=torch.int64),
                }

            targets.append(target)

        return imgs, targets

    def _create_pd_dot_annot(self):
        index, x, y = [], [], []
        for img_n in self.image_names:
            label = imread(os.path.join(self.root, img_n.replace("cell", "dots")))
            label = label[:, :, 0] / 255
            x.extend(np.nonzero(label)[1])
            y.extend(np.nonzero(label)[0])
            index.extend([img_n] * np.count_nonzero(label))

        anns = {'X': x, 'Y': y}
        df_annot = pd.DataFrame(anns, columns=['X', 'Y'], index=index)

        return df_annot


# Debug code
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from utils.misc import normalize
    import random
    from skimage.io import imsave
    from skimage import draw
    import scipy.stats
    from skimage.color import gray2rgb

    ROOT = "/home/luca/luca-cnr/mnt/datino/VGG_cells"
    NUM_TRAIN_SAMPLE = 32
    assert NUM_TRAIN_SAMPLE < 51, "The maximum number of train samples is 50"
    output_folder = "output/gt/cells/"
    ann_type = "det"
    device = "cpu"
    RED = [255, 0, 0]

    def is_empty(l):
        return all(is_empty(i) if isinstance(i, list) else False for i in l)


    img_names = [img_name for img_name in os.listdir(ROOT) if img_name.endswith("cell.png")]
    num_train_sample = NUM_TRAIN_SAMPLE
    num_val_sample = 100 - num_train_sample
    indexes = list(range(0, len(img_names)))  # the dataset contains 200 images; 100 should be for test
    random.shuffle(indexes)
    train_indexes = indexes[0:num_train_sample]
    train_img_names = [img_names[i] for i in train_indexes]
    val_indexes = indexes[num_train_sample:num_val_sample]
    val_img_names = [img_names[i] for i in val_indexes]
    test_img_names = list(set(img_names) - set(train_img_names + val_img_names))

    if ann_type == "den":
        from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose
        transforms = Compose([ToTensor(), RandomHorizontalFlip()])
    elif ann_type == "det":
        from det_transforms import ToTensor, RandomHorizontalFlip, Compose
        transforms = Compose([RandomHorizontalFlip(), ToTensor()])

    dataset = VGGCellsDataset(
        ROOT,
        transforms=transforms,
        in_memory=False,
        image_names=train_img_names,
        ann_type=ann_type,
    )
    if ann_type == "den":
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    elif ann_type == "det":
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=dataset.custom_collate_fn)

    progress = tqdm(dataloader, desc='TRAIN', leave=False)
    for i, sample in enumerate(progress):
        input_and_target, _, _, _, image_id = sample

        if ann_type == "den":
            images, targets = input_and_target.split(1, dim=1)
            for img, dmap, img_id in zip(images, targets, image_id):
                img_name = img_id.rsplit(".", 1)[0] + "_{}.png".format(i)
                dmap_name = img_id.rsplit(".", 1)[0] + "_{}_dmap.png".format(i)

                image = gray2rgb(img.squeeze().cpu().numpy()) * 255
                imsave(os.path.join(output_folder, img_name), image)
                imsave(os.path.join(output_folder, dmap_name), normalize(dmap.squeeze().cpu().numpy()))
        elif ann_type == "det":
            imgs, targets = dataset.build_coco_compliant_batch(input_and_target)
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            for img, target, img_id in zip(imgs, targets, image_id):
                img_name = img_id.rsplit(".", 1)[0] + "_withBBs.png"
                bboxes = target['boxes'].tolist()

                image = gray2rgb(img.squeeze().cpu().numpy())*255
                for bb in bboxes:
                    rs, cs = int(bb[1]), int(bb[0])
                    re, ce = int(bb[3]), int(bb[2])
                    rr, cc = draw.rectangle_perimeter(start=(rs, cs), end=(re, ce), shape=image.shape)
                    image[rr, cc] = RED
                imsave(os.path.join(output_folder, img_name), image)


