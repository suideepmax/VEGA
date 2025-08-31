import cv2 as cv


#reading an image
img = cv.imread('Photos/cat.png')

cv.imshow('Cat', img)
cv.waitKey(0)

#reading a video
capture = cv.VideoCapture('Videos/dog.mp4')

while True:
    isTrue, frame = capture.read() #reads the video frame by frame
    if not isTrue:
        break
    cv.imshow('Dog', frame) #displays the frame
     #waits for 20ms before displaying the next frame
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()