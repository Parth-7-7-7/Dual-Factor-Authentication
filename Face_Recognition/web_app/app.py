import streamlit as st
import cv2
import numpy as np
import imutils
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import time
import os

# Set minimum probability threshold for verification
PROBA_THRESHOLD = 0.6  # Increase to 0.7 or 0.8 for stricter verification

# Load models
@st.cache_resource(show_spinner=True)
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    protoPath = os.path.join(base_dir, "..", "face_detection_model", "deploy.prototxt")
    modelPath = os.path.join(base_dir, "..", "face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel")
    embedderPath = os.path.join(base_dir, "..", "face_recognition_model", "openface_nn4.small2.v1.t7")
    recognizerPath = os.path.join(base_dir, "..", "output", "recognizer.pickle")
    lePath = os.path.join(base_dir, "..", "output", "le.pickle")

    for path in [protoPath, modelPath, embedderPath, recognizerPath, lePath]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required model file: {path}")

    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    embedder = cv2.dnn.readNetFromTorch(embedderPath)
    recognizer = pickle.loads(open(recognizerPath, "rb").read())
    le = pickle.loads(open(lePath, "rb").read())
    return detector, embedder, recognizer, le

detector, embedder, recognizer, le = load_models()


class FaceRecognizer(VideoTransformerBase):
    def __init__(self):
        self.verified_name = None
        self.last_verified_time = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        img = imutils.resize(img, width=600)
        (h, w) = img.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        verified_name = None
        max_proba = 0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = img[startY:endY, startX:endX]
                if face.shape[0] < 20 or face.shape[1] < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                                 (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                if proba > max_proba:
                    max_proba = proba
                    if proba >= PROBA_THRESHOLD:
                        verified_name = name

                label = name if proba >= PROBA_THRESHOLD else "Unknown"
                text = f"{label}: {proba * 100:.2f}%"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

        if verified_name:
            cv2.putText(img, f"Verified: {verified_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            self.verified_name = verified_name
            self.last_verified_time = time.time()
        else:
            cv2.putText(img, "Not Verified", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return img


def recognize_image(image):
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    verified_name = None
    max_proba = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            if proba > max_proba:
                max_proba = proba
                if proba >= PROBA_THRESHOLD:
                    verified_name = name

            label = name if proba >= PROBA_THRESHOLD else "Unknown"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, f"{label}: {proba * 100:.2f}%", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

    if verified_name:
        cv2.putText(image, f"Verified: {verified_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        cv2.putText(image, "Not Verified", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return image, verified_name


def main():
    st.title("Face Recognition with Webcam and Image Upload")
    st.write("### Choose an option below:")

    option = st.radio("", ("Webcam Face Recognition", "Upload Image for Recognition"))

    verified_name = None

    if option == "Webcam Face Recognition":
        st.write("Starting webcam...")

        result = webrtc_streamer(
            key="face-recognizer", 
            video_transformer_factory=FaceRecognizer,
            media_stream_constraints={"video": True, "audio": False},
        )

        if result.video_transformer and result.video_transformer.verified_name:
            verified_name = result.video_transformer.verified_name

    elif option == "Upload Image for Recognition":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            result_img, verified_name = recognize_image(img)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Recognition Result", use_column_width=True)

    if verified_name:
        st.success(f"Identity Verified: {verified_name}. Redirecting to OTP verification...")
        st.markdown(f"""
        <meta http-equiv="refresh" content="2; url=http://127.0.0.1:5000/?verified_name={verified_name}" />
        If you are not redirected, <a href="http://127.0.0.1:5000/?verified_name={verified_name}">click here</a>.
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
