import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

print("[INFO] Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("face_recognition_model/openface_nn4.small2.v1.t7")

print("[INFO] Loading face recognition model and label encoder...")
with open("output/recognizer.pickle", "rb") as f:
    recognizer = pickle.load(f)
with open("output/le.pickle", "rb") as f:
    le = pickle.load(f)

print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

fps = FPS().start()

while True:
    frame = vs.read()
    if frame is None:
        print("[WARNING] No frame captured from video stream")
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0),
                                      swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Debug info
    print(f"[DEBUG] Frame shape: {frame.shape}, Mean pixel value: {frame.mean():.2f}")
    print(f"[DEBUG] Number of detections: {detections.shape[2]}")

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Clamp bounding box coordinates to frame dimensions
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w - 1, endX)
        endY = min(h - 1, endY)

        face = frame[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        if fW < 20 or fH < 20:
            continue

        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                         (96, 96), (0, 0, 0),
                                         swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        text = f"{name}: {proba * 100:.2f}%"
        y = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    fps.update()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

fps.stop()
print(f"[INFO] Elapsed time: {fps.elapsed():.2f}")
print(f"[INFO] Approx. FPS: {fps.fps():.2f}")

cv2.destroyAllWindows()
vs.stop()
