import cv2 as cv
import numpy as np

img = cv.imread('Photos/cat.png')
cv.imshow('Cat', img)

blank = np.zeros(img.shape[:2], dtype=np.uint8) #image of mask has the same height and width as the original image
cv.imshow('Blank', blank)

mask = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1) #creates a circular mask
cv.imshow('Mask', mask)

masked = cv.bitwise_and(img, img, mask=mask) #using bitwise AND to apply the mask
cv.imshow('Masked', masked)

cv.waitKey(0)