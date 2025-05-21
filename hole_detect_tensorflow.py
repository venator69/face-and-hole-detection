import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# ==== Setup ====
print(f"TensorFlow version: {tf.__version__}")
CATEGORIES = ['normal', 'potholes']
DATA_DIR = 'potholedataset'  # Folder must contain 'normal/' and 'potholes/'

# ==== Load and Preprocess Dataset ====
features, labels = [], []

def load_dataset():
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        label = CATEGORIES.index(category)
        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            try:
                img = cv.imread(img_path)
                img = cv.resize(img, (64, 64))
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = cv.GaussianBlur(img, (5, 5), 0)
                img = cv.Canny(img, 100, 200)
                features.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

load_dataset()

features = np.array(features).reshape(-1, 64, 64, 1).astype('float32') / 255.0
labels = to_categorical(labels, num_classes=2)

# ==== Train/Test Split ====
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# ==== Build CNN Model ====
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

# ==== Compile and Train Model ====
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# ==== Save the Model ====
model.save("pothole_detector.h5")
print("✅ Model saved as 'pothole_detector.h5'")

# ==== Plot Training History ====
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==== Real-Time Webcam Detection ====
print("🚦 Starting real-time pothole detection. Press 'q' to quit.")
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Preprocess the frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, (64, 64))
    blurre

