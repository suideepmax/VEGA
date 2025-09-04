import cv2 as cv

img = cv.imread('Photos/man.png')
cv.imshow('Image', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml') # Load Haar Cascade

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3) # Detect faces

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv.imshow('Detected Faces', img)
cv.waitKey(0)
