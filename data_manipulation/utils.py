import numpy as np
import numpy.ma as ma
from tqdm import tqdm


from py360convert import e2c
import py360convert

def convert_img2cub(img, mode='nearest', format='horizon'):
    if len(img.shape) == 2:
        img = img[..., None]
    face_w = round((np.sqrt(3) / 6) * img.shape[1])  # https://krpano.com/forum/wbb/index.php?page=Thread&threadID=2018
    out = e2c(img, face_w=face_w, mode=mode, cube_format=format)
    return out


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

def get_masked_array(X, Y, Z):
    print(X.shape)

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

    mask = np.zeros_like(X)
    mask[np.isnan(angle_diff)] = 1

    return angle_diff, mask


def calculate_missing_pxls(X,  angles, mask):
    masked_array = ma.array(data= angles, mask = mask)
    global_steps = masked_array.mean(axis=1)
    print(global_steps.shape)

    global_steps = global_steps.filled(global_steps.mean())
    print(global_steps[800], type(global_steps[800]))
    print(global_steps[1500], type(global_steps[1500]))

    print(f'Total angles sum: {global_steps.sum()}, missing angles {np.pi - global_steps.sum()}')
    print(f'Missing pixels: {(np.pi - global_steps.sum()) /global_steps.mean()}')
    print(f'Image pixels width/2: {X.shape[1]//2},  Corrected Image pixels width: {X.shape[0] + (np.pi - global_steps.sum()) / global_steps.mean()}')

    pixels_to_add = int((np.pi - global_steps.sum()) /global_steps.mean())

    return pixels_to_add


def construct_corrected_image(pixels_to_add, img):
    img = np.pad(img, ((0,pixels_to_add),(0,0)), 'constant')

    return img