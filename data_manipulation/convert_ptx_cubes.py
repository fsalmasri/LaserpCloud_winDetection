import os
from os.path import exists, join
import numpy as np
import numpy.ma as ma
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_manipulation.cloud_utils import cloud2im
from py360convert import e2c


def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def get_Interquartile_range(angles_steps):
    values = angles_steps[angles_steps != 0]
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)

    return upper, lower

X, Y, Z = np.load('XYZ.npy')
img = np.load('intensity.npy')

# plt.figure()
# plt.imshow(img)
print(X.shape)

def get_masked_array():
    angle_diff = np.zeros_like(X)
    angle_diff[:,:] = np.nan

    for j in tqdm(range(0, X.shape[1], 1)):
        XYZ = np.vstack((X[:,j], Y[:,j], Z[:,j])).T
        XYZRElevAz= appendSpherical_np(XYZ)

        angles_tostudy = XYZRElevAz[:,4]
        angles_steps = angles_tostudy[1:] - angles_tostudy[:-1]

        # find outliers
        upper, lower = get_Interquartile_range(angles_steps)
        for i in range(angles_steps.shape[0]):
            if (angles_steps[i] > lower) and (angles_steps[i] < upper):
                # angle_diff[j, i] = np.nan
                angle_diff[i,j] = angles_steps[i]
        # print(lower, upper, np.count_nonzero(np.isnan(angle_diff[:,j])), np.count_nonzero(~np.isnan(angle_diff[:,j])))


        # angles_toplot = angle_diff[:,j]
        # angles_toplot = angles_toplot[~np.isnan(angles_toplot)]
        # # # plt.boxplot(angle_diff)
        # plt.plot(angles_toplot)
        # plt.show()
        # exit()

    mask = np.zeros_like(X)
    mask[np.isnan(angle_diff)] = 1
    # plt.imshow(mask)
    # plt.show()

    return angle_diff, mask

# angle_diff, mask = get_masked_array()
# np.save('angles.npy', angle_diff)
# np.save('mask.npy', mask)

mask = np.load('mask.npy')
angles = np.load('angles.npy')
masked_array = ma.array(data= angles, mask = mask)

global_steps = masked_array.mean(axis=1)
print(global_steps.shape)

global_steps = global_steps.filled(global_steps.mean())
print(global_steps[800], type(global_steps[800]))
print(global_steps[1500], type(global_steps[1500]))

print(f'Total angles sum: {global_steps.sum()}, missing angles {np.pi - global_steps.sum()}')
print(f'Missing pixels: {(np.pi - global_steps.sum()) /global_steps.mean()}')
print(f'Image pixels width/2: {X.shape[1]//2},  Corrected Image pixels width: {X.shape[0] + (np.pi - global_steps.sum()) / global_steps.mean()}')

# cubes_dir = '/home/feras/Desktop/data/cubes'
# fnames = os.listdir('/home/feras/Desktop/data/PTX/scan_L_ext.ptx')
#
# for fname in fnames:
#     X, Y, Z, img = cloud2im(fname[:-4])


    # if not exists(join(cubes_dir, fname)):
    #     os.makedirs(join(cubes_dir, fname))
    #
    # if len(img.shape) == 2:
    #     img = img[..., None]
    #
    # w = round((np.sqrt(3)/6) * img.shape[1]) #https://krpano.com/forum/wbb/index.php?page=Thread&threadID=2018
    # out = e2c(img, face_w=w, mode='bilinear', cube_format='dict')


    # for f in out:
    #     np.save(join(cubes_dir, fname, f), out[f])


    # exit()