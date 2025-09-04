import cv2 as cv

def rescaleFrame(frame, scale=0.6): #60% of original size
    #works for images, videos and live video
    width = int(frame.shape[1] * scale) #shape[1] is the width
    height = int(frame.shape[0] * scale) #shape[0] is the height
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA) #INTER_AREA is used for shrinking

img = cv.imread('Photos/man.png')
#cv.imshow('Image', img)
resized = rescaleFrame(img)
cv.imshow('Resized Image', resized)

gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray Image', gray)
resized_gray = rescaleFrame(gray)
cv.imshow('Resized Gray Image', resized_gray)

haar_cascade = cv.CascadeClassifier('OpenCV/haar_face.xml') # Load Haar Cascade

faces_rect = haar_cascade.detectMultiScale(resized_gray, scaleFactor=1.1, minNeighbors=4) # Detect faces

print(f'Number of faces found: {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
    cv.rectangle(resized_gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv.imshow('Detected Faces', resized)
cv.waitKey(0)
