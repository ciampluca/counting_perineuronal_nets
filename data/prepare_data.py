import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from skimage import io
import scipy.io
import h5py

DATA_NAMES = ['VGG', 'MBM', 'BCD', 'Adipocyte']



def create_df_ann_from_imgs(data_path, image_paths, data_name):
    """ 
    Create a dataframe containing annotations starting from images containing dots. 
    Useful for VGG, MBM, and Adipocyte
    """

    def load_one_annotation(image_path):
        image_id = image_path.name
        label_map_path = data_path / image_id.replace('cell', 'dots')
        label_map = io.imread(label_map_path)
        if data_name == 'VGG':
            label_map = label_map[:, :, 0]
            y, x = np.where((label_map == 255))
        elif data_name == 'MBM':
            y, x = np.where((label_map == 255))
        elif data_name == 'Adipocyte':
            label_map = label_map[:, :, 0]
            y, x = np.where((label_map == 254))
        
        annot = pd.DataFrame({'Y': y, 'X': x, 'class': 0})
        annot['imgName'] = image_id
        
        return annot

    annot = map(load_one_annotation, image_paths)
    annot = pd.concat(annot, ignore_index=True)
    annot = annot.set_index('imgName')
    
    return annot


def create_df_ann_bcd_data(image_paths, split):
    """ 
    Create a dataframe containing annotations for the BCD dataset. 
    """
    CLASSES = {0: 'positive', 1: 'negative'}
    
    def load_one_annotation(image_path):
        image_id = image_path.name
        
        ann_x, ann_y, num_cl = [], [], []
        for num_class, name_class in CLASSES.items():
            ann_id = image_id.replace('.png', '.h5')
            ann_path = image_path.parent.parent.parent / 'annotations' / split / name_class / ann_id
            ann_h5 = h5py.File(ann_path)
            coordinates = np.asarray(ann_h5['coordinates'])
            for coord in coordinates:
                ann_x.append(coord[0])
                ann_y.append(coord[1])
                num_cl.append(int(num_class))
                
        annot = pd.DataFrame({'Y': np.array(ann_y), 'X': np.array(ann_x), 'class': np.array(num_cl)})
        image_id = image_id.replace('.png', '_cell.png')
        annot['imgName'] = image_id
        
        return annot
        
    annot = map(load_one_annotation, image_paths)
    annot = pd.concat(annot, ignore_index=True)
    annot = annot.set_index('imgName')
    
    return annot



def main(args):
    if args.data_name not in DATA_NAMES:
        print("Not implemented dataset.")
        exit(1)
    
    data_path = Path(__file__).resolve().parent /  Path(args.data_path)
    
    # VGG, MBM
    if args.data_name == 'VGG' or args.data_name == 'MBM':
        image_paths = data_path.glob('*cell.*')
        df_ann = create_df_ann_from_imgs(data_path, image_paths, args.data_name)
        df_ann.to_csv(data_path / 'annotations.csv', encoding='utf-8')
    # BCD
    if args.data_name == 'BCD':    
        image_paths = data_path.glob('*.png')
        split = data_path.name
        df_ann = create_df_ann_bcd_data(image_paths, split)
        dst_fld = data_path.parent.parent.parent / split
        Path(dst_fld / 'imgs').mkdir(parents=True, exist_ok=True)
        df_ann.to_csv(dst_fld / 'annotations.csv', encoding='utf-8')
        image_paths = data_path.glob('*.png')
        for image_path in image_paths:
            dst_path = dst_fld / 'imgs' / image_path.name.replace('.png', '_cell.png')
            dst_path.write_bytes(image_path.read_bytes())
    # Adipocyte
    if args.data_name == 'Adipocyte':
        data_path = Path(data_path / 'Adipocyte_cells')
        image_paths = data_path.glob('*cell.*')
        df_ann = create_df_ann_from_imgs(data_path, image_paths, args.data_name)
        Path(data_path.parent / 'imgs').mkdir(parents=True, exist_ok=True)
        df_ann.to_csv(data_path.parent / 'annotations.csv', encoding='utf-8')
        image_paths = data_path.glob('*cell.*')
        for image_path in image_paths:
            dst_path = data_path.parent / 'imgs' / image_path.name
            dst_path.write_bytes(image_path.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare a dataset to be compliant with the framework', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-name', help="Name of the dataset (can be VGG, MBM, BCD, Adipocyte)")
    parser.add_argument('--data-path', help="Root path of the dataset")
    
    args = parser.parse_args()
    main(args)