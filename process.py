import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from numpy.core.fromnumeric import argsort
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

def otsu_thresh(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def niblack_thresh(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    res = img.copy()
    thresh_niblack = threshold_niblack(img, window_size=25, k=0.8)
    thresh_niblack = cv2.ximgproc.niBlackThreshold(img, maxValue=255, type=cv2.THRESH_BINARY_INV, blockSize=2*11+1, k=-0.2, binarizationMethod=cv2.ximgproc.BINARIZATION_NICK)

    binary_niblack = img > thresh_niblack

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if binary_niblack[i][j]==True:
                res[i][j]=255
            else:
                res[i][j]=0
    return res

def sauvola_thresh(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    res = img.copy()
    thresh_sauvola = threshold_sauvola(img, window_size=25)
    thresh_sauvola = cv2.ximgproc.niBlackThreshold(img, maxValue=255, type=cv2.THRESH_BINARY_INV, blockSize=2*11+1, k=-0.2, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)

    binary_sauvola = img > thresh_sauvola

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if binary_sauvola[i][j]==True:
                res[i][j]=255
            else:
                res[i][j]=0
    return res

def gaussian_adapt(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

def img_blur(img):
    img = cv2.imread(img)
    return cv2.blur(img, (14, 14))

def img_sharpen(img):
    img = cv2.imread(img)
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]])
    sharp_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return sharp_img

def canny_edge(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.Canny(blurred, 30, 150)