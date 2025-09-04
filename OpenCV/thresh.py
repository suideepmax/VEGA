import cv2 as cv

img = cv.imread('Photos/cat.png')
cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Cat', gray)

#simple thresholding
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY) #150 is the threshold value, 255 is the max value to use with the THRESH_BINARY thresholding, cv.THRESH_BINARY is the type of thresholding
cv.imshow('Thresholded', thresh)

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV) #150 is the threshold value, 255 is the max value to use with the THRESH_BINARY thresholding, cv.THRESH_BINARY is the type of thresholding
cv.imshow('Thresholded Inverse', thresh_inv)

#adaptive threshold
thresh_adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
cv.imshow('Thresholded Adaptive', thresh_adaptive)

cv.waitKey(0)