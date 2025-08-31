import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8') #creates a black image
#uint8 means datatype of the image

cv.imshow('Blank', blank)

blank[:] = 0, 255, 0 #BGR value for green
blank[200:300, 300:400] = 255, 0, 0  #BGR value for blue
#changes the whole image to green
cv.imshow('Green', blank)
'''
img = cv.imread('Photos/cat.png')
cv.imshow('Cat', img)'''

cv.rectangle(blank, (0, 0), (250, 250), (255, 0, 0), thickness=2) #draws a rectangle
#use cv.FILLED instead of thickness to fill the rectangle
#(0,0) is the starting point (top-left corner)
cv.imshow('Rectangle', blank)

#cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (255, 0, 0), thickness=cv.FILLED) #draws a filled rectangle
#(blank.shape[1]//2, blank.shape[0]//2) is the ending point (bottom-right corner)

cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0, 0, 255), thickness=3) #draws a circle
#thickness -1 fills the circle
#(blank.shape[1]//2, blank.shape[0]//2) is the center of the circle
#40 is the radius of the circle
cv.imshow('Circle', blank)

cv.line(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (255, 255, 255), thickness=3) #draws a line
#(0,0) is the starting point (top-left corner)
#(blank.shape[1]//2, blank.shape[0]//2) is the ending point (center of the image)
cv.imshow('Line', blank)


cv.putText(blank, 'Hello World!', (225, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 255), thickness=2) #puts text on the image
#(225, 225) is the bottom-left corner of the text

cv.imshow('Text', blank)
cv.waitKey(0)