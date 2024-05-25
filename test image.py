import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def debugShowImage(img):
    cv.imshow('DEBUG', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

'''
contour iamge
contour parchment
step through image contours, match to parchment
find diemensions of parchment, crop image
'''
def main():
    #preprocess input image
    inputimgPath = r"C:\Users\blast\Desktop\TC4-Solver\example research\2024-05-24_14.23.30.png"
    origScreenshot = cv.imread(inputimgPath)
    inputImage = cv.imread(inputimgPath, cv.IMREAD_GRAYSCALE)
    assert inputImage is not None, "file could not be read, check with os.path.exists()"
    ret, inputThresh = cv.threshold(inputImage,100,255,cv.THRESH_BINARY)
    #debugShowImage(inputImage)
    #preprocess minigame area parchment
    pcmnt = cv.imread(r'C:\Users\blast\Desktop\TC4-Solver\images\parchment.png', cv.IMREAD_GRAYSCALE)
    ret, parchThresh = cv.threshold(pcmnt,100,255,cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(parchThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    parchCont = contours[0]
    '''
    parchContImage = cv.drawContours(parchThresh, parchCont, -1, (100,255,100), 3)
    debugShowImage(parchContImage)
    '''

    contours, hierarchy = cv.findContours(inputThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    #debugShowImage(resultImg)

    #512 arbitrary area
    filteredContours = list(filter(lambda c: cv.contourArea(c) > 512, contours))

    minigameCont2 = min(filteredContours, key=lambda c: cv.matchShapes(parchCont, c, 3,0))
    #print(cv.matchShapes(parchCont, minigameCont2, 3,0))
    newIMG = cv.drawContours(origScreenshot.copy(), minigameCont2, -1, (0,255,0), 3)
    newIMG = cv.drawContours(newIMG.copy(), filteredContours, -1, (255,0,0), 3)
    #debugShowImage(newIMG)
    rect = cv.minAreaRect(minigameCont2)
    bound = cv.boxPoints(rect)

    bound = np.intp(bound)
    x, y, w, h = bound[3,0], bound[1,1], bound[1,0], bound[3,1]
    newIMG = cv.drawContours(newIMG.copy(), [bound], -1, (0,0,255), 3)
    debugShowImage(newIMG)
    cropImg = newIMG[y:h, x:w]
    debugShowImage(cropImg)

'''
contour iamge
contour parchment
step through image contours, match to parchment
find diemensions of parchment, crop image




#blur = cv.GaussianBlur(img,(5,5),0)
#debugShowImage(blur)

#debugShowImage(binThresh)



contours, hierarchy = cv.findContours(binThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
minigameArea = max(contours, key=cv.contourArea)
contImg = cv.drawContours(img, minigameArea, -1, (255,255,255), 3)
#debugShowImage(contImg)
ret = cv.matchShapes(parchCont, minigameArea, 1, 0.0)
print(ret)
'''

if __name__ == "__main__":
    main()