import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Photos/cat.png')
cv.imshow('Cat', img)

blank = np.zeros(img.shape[:2], dtype=np.uint8)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Cat', gray)

circle = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1) #creating a circular mask
cv.imshow('Mask', circle)

mask = cv.bitwise_and(gray, gray, mask=circle) #applying the mask to the grayscale image
cv.imshow('Mask Applied on Gray', mask)

#grayscale histogram
# gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256]) #compute a histogram w/o mask
# cv.imshow('Histogram', gray_hist)

gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256]) #compute a histogram w/ mask
cv.imshow('Histogram', gray_hist)

#grayscale histogram
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()

#color histogram
bgr_planes = cv.split(img) #split the image into its color channels
colors = ('b', 'g', 'r')
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
for i, color in enumerate(colors): #looping through the color channels
    #enumerate means to loop through a list and get the index and value
    hist = cv.calcHist([bgr_planes[i]], [0], mask, [256], [0, 256])
    plt.plot(hist, color=color) #plotting the histogram for each color channel
    plt.xlim([0, 256])
plt.show()

cv.waitKey(0)