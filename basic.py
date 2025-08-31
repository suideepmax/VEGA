import cv2 as cv

img = cv.imread('Photos/cat.png')
cv.imshow('Cat', img)

#coverting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

#blur
blur = cv.GaussianBlur(img, (9,9), cv.BORDER_DEFAULT) #kernel size should be odd number
#cv.imshow('Blur', blur)

#edge cascade
canny = cv.Canny(blur, 125, 175) #lower threshold and upper threshold
cv.imshow('Canny Edges', canny)

#dilating the image
dilated = cv.dilate(canny, (7,7), iterations=3) #kernel size and number of iterations
cv.imshow('Dilated', dilated)

#eroding the image
eroded = cv.erode(dilated, (7,7), iterations=3) #kernel size and number of iterations
cv.imshow('Eroded', eroded)

#resizing image
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA) #cv.INTER_AREA is the interpolation method
#cv.INTER_LINEAR is another interpolation method that performs bilinear interpolation
#cv.INTER_CUBIC is another interpolation method that performs bicubic interpolation
#bicubic interpolation is a resampling method that uses the values of the 16 nearest pixels to compute the value of a new pixel
cv.imshow('Resized', resized)

#cropping the image
cropped = img[50:200, 200:400] #y1:y2, x1:x2
cv.imshow('Cropped', cropped) 

cv.waitKey(0)
