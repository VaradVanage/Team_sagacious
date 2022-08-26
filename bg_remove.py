import os
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
segmentor = SelfiSegmentation()
Name="Tomato___Leaf_Mold"

Images = [doc for doc in os.listdir() if doc.endswith('.JPG')]
A=1
for i in Images:
    img = cv2.imread(i)
    img_Out = segmentor.removeBG(img, (255,255,255), threshold=0.99)
    #cv2.imshow('img',img_Out)
    cv2.imwrite(Name+str(A)+".JPG",img_Out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    A=A+1
       
Images1 = [doc for doc in os.listdir() if (doc.endswith('.JPG') and doc[:len(Name)]!=Name)]
for i in Images1:
   os.remove(i)
"""
img = cv2.imread('0a31e630-0d98-416b-b0e4-88a88aad1dc5___RS_HL 9653_new30degFlipLR.JPG')
img_Out = segmentor.removeBG(img, (255,255,255), threshold=0.99)
cv2.imshow('img',img_Out)
cv2.waitKey(0)
cv2.destroyAllWindows()
and doc[:len(Name)]!=Name)

Images1 = [doc for doc in os.listdir() if doc.endswith('.JPG') ]
for i in Images1:
   os.remove(i)
"""