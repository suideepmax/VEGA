import cv2 as cv


#reading an image
img = cv.imread('Photos/cat.png')



def rescaleFrame(frame, scale=0.75): #75% of original size
    #works for images, videos and live video
    width = int(frame.shape[1] * scale) #shape[1] is the width
    height = int(frame.shape[0] * scale) #shape[0] is the height
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA) #INTER_AREA is used for shrinking


def changeRes(width, height):
    #works only for live video
    capture.set(3, width) #3 is the width
    capture.set(4, height) #4 is the height

resized_image = rescaleFrame(img)
cv.imshow('Original', img)
cv.imshow('Cat', resized_image)
cv.waitKey(0)
