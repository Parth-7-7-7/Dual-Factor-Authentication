import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="input image path")
args = vars(ap.parse_args())

# Check if the image file exists
print(f"Trying to read image from: {args['image']}")
if not os.path.isfile(args["image"]):
    print(f"Error: Image file does not exist: {args['image']}")
    exit(1)

# Load serialized face detector from disk
print("Loading Face Detector...")
protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
modelPath = os.path.sep.join(['face_detection_model', "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load serialized face embedding model from disk
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

# Load face recognition model and label encoder
recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
le = pickle.loads(open('output/le.pickle', "rb").read())

# Load the input image
image = cv2.imread(args["image"])
if image is None:
    print(f"Error: Failed to load image from path: {args['image']}")
    exit(1)

# Resize image
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# Construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)

# Detect faces in the image
detector.setInput(imageBlob)
detections = detector.forward()

# Set a threshold to recognize unknown faces
THRESHOLD = 0.5  # Adjust this value if needed

# Loop over detections
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        # Skip small faces
        if fW < 20 or fH < 20:
            continue

        # Create a blob for the face ROI, then obtain the embedding
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                         (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        # Perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]

        # Thresholding logic
        if proba >= THRESHOLD:
            name = le.classes_[j]
        else:
            name = "Unknown"

        # Draw bounding box and label
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
