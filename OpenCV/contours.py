import cv2 as cv
import numpy as np

img = cv.imread('Photos/cat.png')

cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #change to grayscale; different method
cv.imshow('Gray Cat', gray)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

blur = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT) #blur the image; different method

canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny Cat', canny) 

# thresholding
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY) #this will create a thresholded image
#thresholded image means all pixels below 125 will be black and all pixels above 125 will be white

cv.imshow('Thresholded', thresh)

#finding contours
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) #takes edges from the canny image
#RETR_EXTERNAL retrieves only the extreme outer contours
#RETR_TREE retrieves all of the contours and organizes them into a full hierarchy
#RETR_LIST retrieves all of the contours without establishing any hierarchical relationships
#CHAIN_APPROX_NONE retrieves all of the contour points
#CHAIN_APPROX_SIMPLE removes all redundant points and compresses the contour, thereby saving memory
#hierarchies contains information about the image topology
print(len(contours), 'contours found!')

#draw contours on blank image
cv.drawContours(blank, contours, -1, (0, 255, 0), 1) #takes contours from the thresholded image
cv.imshow('Contours', blank)

cv.waitKey(0)

#what is a contour?
# A contour is a curve joining all the continuous points along a boundary that have the same color or intensity.

#how does contour detection work?
# Contour detection works by first converting the image to grayscale, then applying a blur to reduce noise, and finally using the Canny edge detector to find edges in the image. The edges are then used to find contours, which are curves that join all the continuous points along a boundary that have the same color or intensity.

#why are contours useful?
# Contours are useful for shape analysis, object detection and recognition, and image segmentation.

# They help in identifying and locating objects within an image, making them essential for various computer vision tasks.

#what do you mean by binarizing?
# Binarizing is the process of converting an image into a binary image, where the pixels are either black or white. This is typically done by applying a threshold to the image, which separates the foreground (objects of interest) from the background.