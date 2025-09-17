import os
import cv2
import imutils
import pickle
import numpy as np
from imutils import paths

# Set base directory to the location of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct full paths for models and dataset
FACE_DETECTOR_DIR = os.path.join(BASE_DIR, "face_detection_model")
FACE_RECOGNIZER_PATH = os.path.join(BASE_DIR, "face_recognition_model", "openface_nn4.small2.v1.t7")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "output", "embeddings.pickle")

# Load the serialized face detector model
print("[INFO] Loading face detector...")
protoPath = os.path.join(FACE_DETECTOR_DIR, "deploy.prototxt")
modelPath = os.path.join(FACE_DETECTOR_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load the face embedding model
print("[INFO] Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(FACE_RECOGNIZER_PATH)

# Grab image paths and initialize data lists
print("[INFO] Quantifying faces...")
imagePaths = list(paths.list_images(DATASET_DIR))
knownEmbeddings = []
knownNames = []
total = 0

# Loop over images
for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}")
    name = imagePath.split(os.path.sep)[-2]

    # Load image and resize
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False
    )

    # Detect faces
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Proceed if at least one face is detected
    if len(detections) > 0:
        # Get detection with highest confidence
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # Ensure the detection is strong enough
        if confidence > 0.5:
            # Compute bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract face ROI and ensure size
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Construct blob for face and get embedding
            faceBlob = cv2.dnn.blobFromImage(
                face, 1.0 / 255, (96, 96), (0, 0, 0),
                swapRB=True, crop=False
            )
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

# Save embeddings to disk
print(f"[INFO] Serializing {total} encodings...")
data = {"embeddings": knownEmbeddings, "names": knownNames}

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)

with open(EMBEDDINGS_PATH, "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Done.")
