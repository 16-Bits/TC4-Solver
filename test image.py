import cv2 as cv
import numpy as np

debug = False


def debugShowImage(img):  # debug function to show image
    cv.imshow('DEBUG', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def cropMinigame(inputImg):
    # preprocess input screenshot image
    inputColor = inputImg.copy()
    inputGray = cv.cvtColor(inputColor, cv.COLOR_BGR2GRAY)
    ret, inputThresh = cv.threshold(inputGray, 100, 255, cv.THRESH_BINARY)
    # get contours of input screenshot, remove small contours
    contours, hierarchy = cv.findContours(
        inputThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    filteredContours = list(filter(lambda c: cv.contourArea(
        c) > 1024, contours))  # 1024 area is arbitrary
    # preprocess minigame area parchment
    minigameImg = cv.imread(r'images\parchment.png', cv.IMREAD_GRAYSCALE)
    assert minigameImg is not None, "file could not be read, check with os.path.exists()"
    ret, minigameThresh = cv.threshold(minigameImg, 100, 255, cv.THRESH_BINARY)
    # get contours of minigame area
    contours, hierarchy = cv.findContours(
        minigameThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    minigameSourceCont = contours[0]
    # find minigame contour in input image
    minigameInputCont = min(
        filteredContours, key=lambda c: cv.matchShapes(minigameSourceCont, c, 1, 0))
    # get bounding box of minigame contour
    rect = cv.minAreaRect(minigameInputCont)
    bound = np.intp(cv.boxPoints(rect))

    if debug is True:
        # debug visuals of minigame area contours and bounding box
        newIMG = cv.drawContours(
            inputColor.copy(), minigameInputCont, -1, (0, 255, 0), 3)
        newIMG = cv.drawContours(
            newIMG.copy(), filteredContours, -1, (255, 0, 0), 3)
        debugShowImage(newIMG)
        newIMG = cv.drawContours(newIMG.copy(), [bound], -1, (0, 0, 255), 3)
        debugShowImage(newIMG)

    # crop input image based on bounding box info
    x, y, w, h = bound[3, 0], bound[1, 1], bound[1, 0], bound[3, 1]
    cropImg = inputColor[y:h, x:w]
    return cropImg


def main():
    inputImgPath = r"example research\2024-05-15_19.09.52.png"
    screenshotImg = cv.imread(inputImgPath)
    assert screenshotImg is not None, "file could not be read, check with os.path.exists()"
    cropImg = cropMinigame(screenshotImg)
    debugShowImage(cropImg)


if __name__ == "__main__":
    main()
