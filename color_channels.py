import cv2 as cv
import numpy as np

img = cv.imread('Photos/cat.png')

#split the image into its color channels
b, g, r = cv.split(img)
cv.imshow('Blue Channel', b) #blue channel
cv.imshow('Green Channel', g) #green channel
cv.imshow('Red Channel', r) #red channel

print(img.shape) #(height, width, color channels)
print(b.shape) #(height, width)
print(g.shape) #(height, width)
print(r.shape) #(height, width)

#distribution of pixel densities
#lighter means more pixels of that color
#darker means fewer pixels of that color


merged = cv.merge([b, g, r]) #pass a list of channels
#cv.imshow('Merged Image', merged)

blank = np.zeros(img.shape[0:2], dtype=np.uint8)
#why do we use [0:2]? Because we want to create a blank image with the same height and width as the original image, but with only one channel (grayscale).
# cv.imshow('Blank Image', blank)

blue_img = cv.merge([b, blank, blank]) #only blue channel
green_img = cv.merge([blank, g, blank]) #only green channel
red_img = cv.merge([blank, blank, r]) #only red channel

cv.imshow('Blue Image', blue_img)
cv.imshow('Green Image', green_img)
cv.imshow('Red Image', red_img)

cv.waitKey(0)
