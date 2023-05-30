import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from cloud_utils import cloud2im
from py360convert import e2c


# cloud2im('Ulb_L065')

img = np.array(Image.open("data/Ulb_L065.png"))
if len(img.shape) == 2:
    img = img[..., None]

print(img.shape)
w = round((np.sqrt(3)/6) * img.shape[1]) #https://krpano.com/forum/wbb/index.php?page=Thread&threadID=2018
out = e2c(img, face_w=w, mode='bilinear', cube_format='dict')

print(list(out.keys()))

for f in out:
    # np.save(f'data/{f}.npy', out[f])
    # print(out[f].shape)
    plt.imshow(out[f])
    plt.show()