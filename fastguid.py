import time
import random
import cv2

import torch
import numpy as np

from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage.transform import resize

from torch.autograd import gradcheck, Variable

from guided_filter_pytorch.guided_filter import FastGuidedFilter

## GuidedFilter
print('FastGuidedFilter:')
## check forward
# forward on img
rgb = img_as_float(imread('src_img2.jpg'))
gt  = img_as_float(imread('changed.jpg'))
gt = gray2rgb(gt)
# rgb = img_as_float(imread('rgb.jpg'))
# gt  = img_as_float(imread('gt.jpg'))
#import pdb; pdb.set_trace()
x_w, x_h = gt.shape[:2]
w, h = x_w // 8, x_h // 8
lr_rgb = resize(rgb, (w, h), order=0, mode='reflect')
lr_gt = resize(gt, (w, h), order=0, mode='reflect')
inputs = [Variable(torch.from_numpy(i.transpose((2, 0, 1))[None]).float()) for i in [lr_rgb, lr_gt, rgb]]
f = FastGuidedFilter(8, 0)
start_time = time.time()
r = f(*inputs)
end_time = time.time()
print('\tForward on img ...')
print('\t\tTime: {}'.format(end_time-start_time))
r = r.data.cpu().numpy().squeeze().transpose(1, 2, 0)
r = np.asarray(r.clip(0, 1)*255, dtype=np.uint8)
imsave('r_fast.jpg', r)

# backward | check grad
test = gradcheck(FastGuidedFilter(2, random.random()).double(),
                 (Variable(torch.rand((2, 3,  6,  7)).double(), requires_grad=True),
                  Variable(torch.rand((2, 3,  6,  7)).double(), requires_grad=True),
                  Variable(torch.rand((2, 3, 11, 12)).double(), requires_grad=True)), eps=1e-6, atol=1e-4)
print('\tGrad Check Result:', test)