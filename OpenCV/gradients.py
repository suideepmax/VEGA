import cv2 as cv
import numpy as np

img = cv.imread('Photos/cat.png')
cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Cat', gray)

#laplacian
#laplacian means to detect edges in an image in a way that highlights regions of rapid intensity change
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap)) #convert to uint8
cv.imshow('Laplacian', lap)

#sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
# sobel = cv.magnitude(sobelx, sobely)
# sobel = np.uint8(np.absolute(sobel))

#sobel means to detect edges in an image by computing the gradient in the x and y directions
#combined sobel
combined_sobel = cv.bitwise_or(np.uint8(np.absolute(sobelx)), np.uint8(np.absolute(sobely)))
cv.imshow('Combined Sobel', combined_sobel)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)

cv.waitKey(0)