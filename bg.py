# importing os module  
import os

import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

segmentor = SelfiSegmentation()
img = cv2.imread("bg\0a4b3cde-c83a-4c83-b037-010369738152___RS_Late.B 6985_flipLR.JPG")
img_Out = segmentor.removeBG(img, (255,255,255), threshold=0.99)
cv2.imshow('img',img_Out)
cv2.waitKey(0)
cv2.destroyAllWindows()