import numpy as np
from os import listdir
import glob, os
from os.path import join
from PIL import Image
from tqdm import tqdm

from data_manipulation.cloud_utils import cloud2im


ptx_dir = '/home/feras/Desktop/data/PTX'
imgs_dir = '/home/feras/Desktop/data/imgs'
annotations_dir = '/home/feras/Desktop/data/annotations'
XYZ_dir = '/home/feras/Desktop/data/XYZ'
intensity_dir = '/home/feras/Desktop/data/intensity'


fnames = listdir(ptx_dir)

for fname in tqdm(fnames):
    if not os.path.isfile(f'{intensity_dir}/{fname[:-3]}npy'):
        print(f'Loading points in ({fname}) and converts them to image... ')
        XYZ, intensity = cloud2im(join(ptx_dir, fname))

        im = Image.fromarray((intensity * 255).astype(np.uint8))
        im.save(f"{imgs_dir}/{fname[:-3]}png")
        np.save(f'{XYZ_dir}/{fname[:-3]}npy', XYZ)
        np.save(f'{intensity_dir}/{fname[:-3]}npy', intensity)