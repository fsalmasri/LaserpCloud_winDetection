from os.path import join
import numpy as np
import json
import os

import skimage as ski
import matplotlib.pyplot as plt
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None

from utils import convert_img2cub, construct_corrected_image



imgs_dir = '/home/feras/Desktop/data/imgs'
annotations_dir = '/home/feras/Desktop/data/annotations'
XYZ_dir = '/home/feras/Desktop/data/XYZ'
intensity = '/home/feras/Desktop/data/intensity'
cubes_im ='/home/feras/Desktop/data/cubes'
cubes_annotations = '/home/feras/Desktop/data/cubes_annotations'

fnames = [f[:-3] for f in os.listdir(intensity)]
for fname in fnames:

    print(f'Reading image: {fname}')
    # im = np.array(Image.open(join(imgs_dir, fname+'png')))
    im = np.load(join(intensity, fname+'npy'))
    X, Y, Z = np.load(join(XYZ_dir, fname+'npy'))

    with open(join(annotations_dir, fname + 'json')) as f:
        ann = json.load(f)

    assert im.shape == (ann['height'], ann['width'])
    segs = ann['regions']

    im_mask = np.zeros(im.shape)
    for i, s in enumerate(segs):
        seg0_x, seg0_y = s['shape_attributes']['all_points_x'], s['shape_attributes']['all_points_y']

        seg0_y, seg0_x = ski.draw.polygon(seg0_y, seg0_x)
        im_mask[seg0_y, seg0_x] = i + 1


    if im.shape[1] % 2:
        im, im_mask = im[:, :-1], im_mask[:, :-1]

    #calculate missing angles.
    # print('Detect angle steps and predict missing pixels')
    # angles, mask = get_masked_array(X, Y, Z)
    # pixels_to_add = calculate_missing_pxls(X, angles, mask)

    pixels_to_add = (im.shape[1] //2 ) - im.shape[0] # Image im.shape[1] = im.shape[2] /2
    im = construct_corrected_image(pixels_to_add, im)
    im_mask = construct_corrected_image(pixels_to_add, im_mask)

    print('Convert image to cube...')
    im_cubes = convert_img2cub(im, format='dict')
    print('Convert image mask to cube...')
    im_mask_cubes = convert_img2cub(im_mask, format='dict')

    for idx in im_cubes:
        np.save(f'{cubes_im}/{fname}_{idx}.npy', im_cubes[idx])
        np.save(f'{cubes_annotations}/{fname}_{idx}.npy', im_mask_cubes[idx])

    print('\n')

