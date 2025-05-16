# BASIC FACIAL RECOGNITION PROGRAM

import cv2
import sys
import numpy as np

# load the classifier and create a cascade object for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarscascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarscascade_eye.xml")
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarscascade_smile.xml")

# file validation
if faceCascade.empty():
    print('--(!)Error loading face cascade')
    exit()
if eyeCascade.empty():
    print('--(!)Error loading eye cascade')
    exit()
if smileCascade.empty():
    print('--(!)Error loading smile cascade')
    exit()

# sets video source to external webcam (my built in one is missing...)
cap = cv2.VideoCapture(1)

# camera validation
if not cap.isOpened():
    print('--(!)Error opening camera')
    exit()

while True:
    ret, frame = cap.read()

    # converts colored frame to grayscale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face detection
    faces = faceCascade.detectMultiscale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

    for (x, y, w, h) in faces:
        # create rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

        roi_gray = grayscale[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # eye detection
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(15,15),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)

        smiles = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=20,
            minSize=(25, 25)
        )

        # IMPLEMENT EMOTION DETECTION HERE
        # ideally, i want to practice using CNN models

    cv2.imshow('u chose the right career.', frame)

    # exit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# once everything is done, release the capture
cap.release()
cv2.destroyAllWindows()