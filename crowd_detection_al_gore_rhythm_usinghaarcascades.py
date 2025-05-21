import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

capture = cv.VideoCapture(0)  # Use 0 for webcam

if not capture.isOpened():
    print("Error: Cannot open video file.")
else:
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            print("Reached end of video or failed to read frame.")
            break
        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        haar_cascade = cv.CascadeClassifier('haar_face.xml')
        race_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces_rect = haar_cascade.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=3)  
        
        cv.imshow('Faces Detected', frame)  


        for (x, y, w, h) in faces_rect:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv.putText(frame, 'Face', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv.imshow('Detected Faces', frame)
        # Press 'd' to exit the loop

   


        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()