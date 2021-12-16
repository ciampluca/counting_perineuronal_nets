from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from datasets.patched_datasets import RandomAccessImageDataset, RandomAccessMultiImageDataset


class PerineuronalNetsRankDataset(Dataset):

    def __init__(
        self,
        root='data/perineuronal-nets/test',
        patch_size=64,
        split='all',
        split_seed=87,
        split_type='cell',
        random_offset=0,
        neg_fraction=0,
        mode='tuples',
        n_tuples=1000,
        transforms=None,
        max_cache_mem=None
    ):

        assert mode in ('tuples', 'patches'), f'Unsupported mode: {mode}'

        self.root = Path(root)
        self.patch_size = patch_size
        self.split = split
        self.split_seed = split_seed
        self.random_offset = random_offset
        self.neg_fraction = neg_fraction
        self.mode = mode
        self.n_tuples = n_tuples

        self.annot = pd.read_csv(self.root / 'annotations.csv')
        self.annot['agreement'] = self.annot.loc[:, 'AV':'VT'].sum(axis=1)

        self.random_state = np.random.RandomState(seed=split_seed)

        self._sample_negatives()

        if split != 'all':
            if split_type == 'cell':
                train, valtest = train_test_split(self.annot, test_size=0.30, stratify=self.annot.agreement, random_state=self.random_state)
                val, test = train_test_split(valtest, test_size=0.50, stratify=valtest.agreement, random_state=self.random_state)
            elif split_type == 'image':
                images = self.annot.imgName.unique()
                train, valtest = train_test_split(images, test_size=0.50, random_state=self.random_state)
                val, test = train_test_split(valtest, test_size=0.50, random_state=self.random_state)
                train = self.annot[self.annot.imgName.isin(train)].reset_index()
                val = self.annot[self.annot.imgName.isin(val)].reset_index()
                test = self.annot[self.annot.imgName.isin(test)].reset_index()
            self.annot = train if split == 'train' else val if split == 'validation' else test
        
        # annotations should be sorted by imgName to align to the patches indices
        self.annot = self.annot.sort_values('imgName', ascending=True, ignore_index=True)
        
        image_groups = self.annot.groupby('imgName')
        image_paths = [(self.root / 'fullFramesH5' / i).with_suffix('.h5') for i in image_groups.groups.keys()]
        image_annots = [annot[['Y', 'X']].values for _, annot in image_groups]

        if self.random_offset:
            image_annots = [a + self.random_state.randint(-self.random_offset, self.random_offset, size=a.shape)
                            for a in image_annots]

        kwargs = dict(
            patch_size=self.patch_size,
            transforms=transforms,
            max_cache_mem=max_cache_mem
        )
        datasets = [RandomAccessImageDataset(p, a, **kwargs) for p, a in zip(image_paths, image_annots)]
        self.patches = RandomAccessMultiImageDataset(datasets)

        # generate tuples
        if self.mode == 'tuples':
            self.generate_tuples()
    
    def _sample_negatives(self):
        if not self.neg_fraction:
            return
        
        def _add_negatives(a):
            n_neg = int(self.neg_fraction * len(a))

            pos_yx = a[['Y', 'X']].values
            neg_yx = np.empty((0, 2), dtype=self.annot.X.dtype)
            threshold = 1.2 * self.patch_size  # negative can be a little overlapped with positive

            while len(neg_yx) < n_neg:
                n_samples = n_neg - len(neg_yx)
                X = self.random_state.randint(2000, size=n_samples)  # TODO hardcoded width=2000
                Y = self.random_state.randint(2000, size=n_samples)  # TODO hardcoded height=2000
                yx = np.stack((Y, X), axis=1)

                distance_matrix = cdist(yx, pos_yx, 'euclidean')
                good = (distance_matrix > threshold).all(axis=1)
                neg_yx = np.vstack((neg_yx, yx[good]))
            
            neg_annot = pd.DataFrame(neg_yx, columns=['Y','X'])
            neg_annot['agreement'] = 0
            neg_annot['imgName'] = a.name

            return pd.concat((a, neg_annot), ignore_index=True)

        self.annot = self.annot.groupby('imgName').apply(_add_negatives).reset_index(drop=True)

    def __len__(self):
        if self.mode == 'tuples':
            return self.n_tuples

        if self.mode == 'patches':
            return len(self.patches)
    
    def __getitem__(self, index):
        if self.mode == 'tuples':
            sample = [self.patches[i] for i in self.tuples[index]]
        elif self.mode == 'patches':
            sample = self.patches[index], self.annot.iloc[index].agreement
        
        return sample

    def generate_tuples(self):
        if self.mode != 'tuples':
            return
            
        tuples = {
            key: group.sample(n=self.n_tuples, replace=True, random_state=self.random_state).index
            for key, group in self.annot.groupby('agreement')
        }

        tuples = [tuples[k] for k in sorted(tuples.keys())]  # sort groups by key
        tuples = tuple(zip(*tuples))  # a collection of tuples
        self.tuples = tuples
    
    def __str__(self):
        if self.mode == 'tuples':
            s = f'{self.__class__.__name__}: ' \
                f'{len(self.tuples)} tuples from ' \
                f'{len(self.patches)} patches of ' \
                f'{self.patches.num_images()} image(s)'
                
        elif self.mode == 'patches':
            s = f'{self.__class__.__name__}: ' \
                f'{len(self.patches)} patches of ' \
                f'{self.patches.num_images()} image(s)'

        return s
