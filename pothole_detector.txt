import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# === Load trained model ===
model = load_model('pothole_detector.h5')
print("✅ Loaded model: pothole_detector.h5")

# === Define label categories ===
CATEGORIES = ['normal', 'potholes']

# === Open webcam ===
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("🚦 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Process frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, (64, 64))
    blurred = cv.GaussianBlur(resized, (5, 5), 0)
    edges = cv.Canny(blurred, 100, 200)
    normalized = edges.astype('float32') / 255.0
    input_img = normalized.reshape(1, 64, 64, 1)

    # Prediction
    prediction = model.predict(input_img, verbose=0)
    class_index = np.argmax(prediction)

    # If pothole detected, draw red rectangle
    if CATEGORIES[class_index] == 'potholes':
        h, w = frame.shape[:2]
        cv.rectangle(frame, (w//4, h//4), (w*3//4, h*3//4), (0, 0, 255), 2)
        cv.putText(frame, "POTHOLE DETECTED", (w//4, h//4 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show frame
    cv.imshow("Pothole Detector", frame)

    # Exit with 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv.destroyAllWindows()
