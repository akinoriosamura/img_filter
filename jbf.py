import cv2
import numpy as np
import random
import math

src = cv2.imread('./changed.jpg', cv2.IMREAD_COLOR)#.astype(float)
joint = cv2.imread('./src_img.jpg', cv2.IMREAD_COLOR)#.astype(float)
img_shape = src.shape
joint = cv2.resize(joint, (img_shape[1], img_shape[0]))
dst = cv2.ximgproc.jointBilateralFilter(joint,src,1000,100,100)
# dst = cv.ximgproc.jointBilateralFilter(src,src,33,2,0) #采用src作为joint

cv2.imwrite("img.jpg",src)
cv2.imwrite("joint.jpg",joint)
cv2.imwrite("dst.jpg",dst)