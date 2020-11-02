import cv2
import numpy as np
import random
import math

src = cv2.imread('./prediction_mask_color2.jpg', cv2.IMREAD_GRAYSCALE)
import pdb; pdb.set_trace()
# th, src = cv2.threshold(src, 128, 255, cv2.THRESH_OTSU)
src[src!=76] = 0
src[src==76] = 255
cv2.imwrite("changed.jpg",src)