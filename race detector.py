import cv2 as cv
import numpy as np

# Ethnicity label map
ethnicity_labels = {
    0: "African American",
    1: "East Asian",
    2: "Caucasian Latin",
    3: "Asian Indian"
}

# Open webcam
capture = cv.VideoCapture(0)

# Load face detector and race classifier
face_cascade = cv.CascadeClassifier('haar_face.xml')
race_classifier = cv.CascadeClassifier('finalTrain.xml')  # VMER-based XML

if not capture.isOpened():
    print("Error: Cannot open webcam.")
else:
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            print("Failed to read frame.")
            break

        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=3)

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract face ROI
            face_roi = grayscale[y:y+h, x:x+w]

            # Use race classifier on the face region
            race_preds = race_classifier.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)

            # Assuming finalTrain.xml returns different boxes based on ethnicity class
            ethnicity = "Unknown"
            for (rx, ry, rw, rh) in race_preds:
                # Here, the trick is: we assume the label index is somehow embedded in the detection process
                # Since Haar cascades can't assign multiple classes, this logic only works
                # if you have multiple classifiers or a workaround that sets `ethnicity` from prediction
                # Since OpenCV's `CascadeClassifier` can't provide label IDs, this won't work unless hacked

                # PLACEHOLDER logic — you'll need to modify this based on how you trained/exported the .xml
                predicted_label = 1  # <-- REPLACE with real logic from classifier output
                ethnicity = ethnicity_labels.get(predicted_label, "Unknown")
                break  # stop after first detection

            # Display label
            cv.putText(frame, ethnicity, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        cv.imshow("Ethnicity Detection", frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()
