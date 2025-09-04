import cv2 as cv

img = cv.imread('Photos/cat.png')
cv.imshow('Cat', img)

#averaging
blurred = cv.blur(img, (3, 3)) #higher kernel size means more blurring
cv.imshow('Average Blurred', blurred)

#gaussian blur
gauss = cv.GaussianBlur(img, (3, 3), 0) #0 means no additional Gaussian kernel standard deviation
cv.imshow('Gaussian Blurred', gauss)

#median blur
median = cv.medianBlur(img, 3) #more effective at removing salt-and-pepper noise
#kernel size is just an integer 
#used in advanced image processing
cv.imshow('Median Blurred', median)

#bilateral filter
#used in advanced image processing
bilateral = cv.bilateralFilter(img, 5, 75, 75) #detailed edges while reducing noise
#larger values of sigmaColor and sigmaSpace lead to more blurring
#sigmaColor controls the filter's sensitivity to color differences
#sigmaSpace controls the filter's sensitivity to spatial differences
cv.imshow('Bilateral Blurred', bilateral)

cv.waitKey(0)