import os
from os.path import exists, join
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from data_manipulation.cloud_utils import cloud2im
from py360convert import e2c


cubes_dir = '/home/feras/Desktop/data/cubes'
fname = 'Ulb_L065'
img = cloud2im(fname)

if not exists(join(cubes_dir, fname)):
    os.makedirs(join(cubes_dir, fname))
print(img.shape)

exists()
# if len(img.shape) == 2:
#     img = img[..., None]

# w = round((np.sqrt(3)/6) * img.shape[1]) #https://krpano.com/forum/wbb/index.php?page=Thread&threadID=2018
# out = e2c(img, face_w=w, mode='bilinear', cube_format='dict')
#
# print(list(out.keys()))
#
# for f in out:
#     # np.save(f'data/{f}.npy', out[f])
#     # print(out[f].shape)
#     plt.imshow(out[f])
#     plt.show()