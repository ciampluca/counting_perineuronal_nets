import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from skimage import io
import scipy.io

DATA_NAMES = ['VGG', 'MBM', 'nuclei']



def create_df_ann_from_imgs(data_path, image_paths, data_name):
    """ 
    Create a dataframe containing annotations starting from images containing dots. 
    Useful for VGG and MBM
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
        
        annot = pd.DataFrame({'Y': y, 'X': x, 'class': 0})
        annot['imgName'] = image_id
        
        return annot

    annot = map(load_one_annotation, image_paths)
    annot = pd.concat(annot, ignore_index=True)
    annot = annot.set_index('imgName')
    
    return annot


def create_df_ann_nuclei_data(image_paths):
    """ 
    Create a dataframe containing annotations for the Nuclei dataset. 
    """
    CLASSES = {0: 'epithelial', 1: 'fibroblast', 2: 'inflammatory', 3: 'others'}
    
    def load_one_annotation(image_path):
        image_id = image_path.name
        
        ann_x, ann_y, num_cl = [], [], []
        for num_class, name_class in CLASSES.items():
            ann_id = image_id.replace('.bmp', '_{}.mat'.format(name_class))
            ann_path = image_path.parent / ann_id
            ann_mat = scipy.io.loadmat(ann_path)
            for ann in ann_mat['detection']:
                ann_x.append(int(round(ann[0])))
                ann_y.append(int(round(ann[1])))
                num_cl.append(int(num_class))
                
        annot = pd.DataFrame({'Y': np.array(ann_y), 'X': np.array(ann_x), 'class': np.array(num_cl)})
        image_id = image_id.replace('.bmp', '_cell.bmp')
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
    # Nuclei
    if args.data_name == 'nuclei':
        data_path = data_path / 'CRCHistoPhenotypes_2016_04_28' / 'Classification'
        image_paths = data_path.glob('**/*.bmp')
        df_ann = create_df_ann_nuclei_data(image_paths)
        dst_fld = data_path.parent.parent
        Path(dst_fld / 'imgs').mkdir(parents=True, exist_ok=True)
        df_ann.to_csv(dst_fld / 'annotations.csv', encoding='utf-8')
        image_paths = data_path.glob('**/*.bmp')
        for image_path in image_paths:
            dst_path = dst_fld / 'imgs' / image_path.name.replace('.bmp', '_cell.bmp')
            dst_path.write_bytes(image_path.read_bytes())
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare a dataset to be compliant with the framework', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-name', help="Name of the dataset (can be VGG, MBM, nuclei)")
    parser.add_argument('--data-path', help="Root path of the dataset")
    
    args = parser.parse_args()
    main(args)