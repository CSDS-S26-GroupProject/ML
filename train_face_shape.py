import cv2
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# INITIAL SETUP
# =========================

print("Script started")

MODEL_PATH = "face_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# =========================
# FUNCTIONS
# =========================

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])


def compute_features(landmarks):
    # Landmark indices
    FOREHEAD_TOP = 10
    CHIN = 152

    LEFT_FOREHEAD = 127
    RIGHT_FOREHEAD = 356

    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454

    LEFT_JAW = 234
    RIGHT_JAW = 454

    LEFT_CHIN = 172
    RIGHT_CHIN = 397

    LEFT_EYE = 33
    RIGHT_EYE = 263

    # Basic distances
    face_height = np.linalg.norm(
        landmarks[FOREHEAD_TOP][:2] - landmarks[CHIN][:2]
    )

    face_width = np.linalg.norm(
        landmarks[LEFT_CHEEK][:2] - landmarks[RIGHT_CHEEK][:2]
    )

    forehead_width = np.linalg.norm(
        landmarks[LEFT_FOREHEAD][:2] - landmarks[RIGHT_FOREHEAD][:2]
    )

    jaw_width = np.linalg.norm(
        landmarks[LEFT_JAW][:2] - landmarks[RIGHT_JAW][:2]
    )

    chin_width = np.linalg.norm(
        landmarks[LEFT_CHIN][:2] - landmarks[RIGHT_CHIN][:2]
    )

    eye_width = np.linalg.norm(
        landmarks[LEFT_EYE][:2] - landmarks[RIGHT_EYE][:2]
    )

    # Jaw angle proxy (angle at chin)
    jaw_vector_left = landmarks[LEFT_JAW][:2] - landmarks[CHIN][:2]
    jaw_vector_right = landmarks[RIGHT_JAW][:2] - landmarks[CHIN][:2]

    cosine_angle = np.dot(jaw_vector_left, jaw_vector_right) / (
        np.linalg.norm(jaw_vector_left) * np.linalg.norm(jaw_vector_right)
    )

    jaw_angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    # Ratios (scale-invariant)
    return [
        face_width / face_height,
        forehead_width / face_width,
        jaw_width / face_width,
        chin_width / jaw_width,
        eye_width / face_width,
        forehead_width / jaw_width,
        jaw_width / face_height,
        jaw_angle
    ]


# =========================
# DATASET BUILDING
# =========================

dataset_path = "dataset"
data = []
labels = []

print("Scanning dataset directory...")

for label in sorted(os.listdir(dataset_path)):
    class_dir = os.path.join(dataset_path, label)

    if not os.path.isdir(class_dir):
        continue

    print(f"Processing class: {label}")

    image_files = sorted(os.listdir(class_dir))[:50]  # limit for debugging

    for img_file in image_files:
        img_path = os.path.join(class_dir, img_file)

        landmarks = extract_landmarks(img_path)
        if landmarks is None:
            continue

        features = compute_features(landmarks)
        data.append(features)
        labels.append(label)

print("Finished dataset processing")

# =========================
# FINAL DATA ARRAYS
# =========================

X = np.array(data)
y = np.array(labels)

print("Final dataset size:", X.shape)
print("Labels count:", len(y))


# =========================
# MODEL TRAINING
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Top-1 classification accuracy (diagnostic only): {accuracy:.3f}")

# =========================
# PROBABILISTIC INTERPRETATION
# =========================

print("\nSample probabilistic geometry profiles (top-2):\n")

probs = model.predict_proba(X_test)
classes = model.classes_

for i in range(5):
    top = np.argsort(probs[i])[::-1][:2]
    print(
        f"Primary: {classes[top[0]]} ({probs[i][top[0]]:.2f}) | "
        f"Secondary: {classes[top[1]]} ({probs[i][top[1]]:.2f})"
    )