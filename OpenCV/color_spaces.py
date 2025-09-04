import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('Photos/cat.png')
cv.imshow('Cat', img)

#BGR to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Cat', gray)

#BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) #hsv is the Hue, Saturation, and Value color space
cv.imshow('HSV Cat', hsv)
#HSV to BGR uses cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
# cv.imshow('BGR Cat from HSV', cv.cvtColor(hsv, cv.COLOR_HSV2BGR))

#BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2Lab) #Lab is a color space that includes all perceivable colors, which means that its gamut exceeds those of the RGB and CMYK color models
cv.imshow('LAB Cat', lab)
# cv.imshow('BGR Cat from LAB', cv.cvtColor(lab, cv.COLOR_Lab2BGR)) #reverse

#BGR to RGB
# plt.imshow(img) #matplotlib uses RGB color space by default
# plt.show()

#convert BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) #color code conversion
cv.imshow('RGB Cat', rgb)
plt.imshow(rgb)
plt.show()

cv.waitKey(0)