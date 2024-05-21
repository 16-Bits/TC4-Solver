import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def debugShowImage(img):
    cv.imshow('DEBUG', img)
    cv.waitKey(0)

img = cv.imread(r'C:\Users\blast\Desktop\TC4 Solver\example research\2024-05-15_19.09.52.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
#blur = cv.GaussianBlur(img,(5,5),0)
#debugShowImage(blur)
ret, binThresh = cv.threshold(img,100,255,cv.THRESH_BINARY)
#debugShowImage(binThresh)

pcmnt = cv.imread(r'C:\Users\blast\Desktop\TC4 Solver\images\parchment.png', cv.IMREAD_GRAYSCALE)
ret, parchThresh = cv.threshold(pcmnt,100,255,cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(parchThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
parchCont = contours[0]
contParch = cv.drawContours(parchThresh, parchCont, -1, (100,255,100), 3)
#debugShowImage(contParch)

contours, hierarchy = cv.findContours(binThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
minigameArea = max(contours, key=cv.contourArea)
contImg = cv.drawContours(img, minigameArea, -1, (255,255,255), 3)
#debugShowImage(contImg)
'''
contour iamge
contour parchment
step through image contours, match to parchment
find diemensions of parchment, crop image
'''
ret = cv.matchShapes(parchCont, minigameArea, 1, 0.0)
print(ret)


