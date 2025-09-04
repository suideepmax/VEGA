import cv2 as cv
import numpy as np

img = cv.imread('Photos/cat.png')
cv.imshow('Cat', img)

#translation
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]]) #transformation matrix
    dimensions = (img.shape[1], img.shape[0])
    dst = cv.warpAffine(img, transMat, dimensions) #take the matrix and apply it to the image
    return dst
# -x -> left
# -y -> up
# +x -> right
# +y -> down

translated = translate(img, -100, 100)
cv.imshow('Translated', translated)


#rotation
def rotate(img, angle, rotPoint=None):
    (height,width) = img.shape[:2] #shape returns (height, width, color channels)
    if rotPoint is None: #if no rotation point is provided
        rotPoint = (width // 2, height // 2) #center of the image

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0) #rotation matrix
    dimensions = (width, height)
    dst = cv.warpAffine(img, rotMat, dimensions) #wrapAffine means to apply an affine transformation to an image
    return dst
 # +angle -> rotate counter-clockwise
 # -angle -> rotate clockwise
 
rotated = rotate(img, 45)
cv.imshow('Rotated', rotated)

#flipping
flipped = cv.flip(img, 1) #1 means flip horizontally, 0 means flip vertically, -1 means flip both
cv.imshow('Flipped', flipped)

cv.waitKey(0)