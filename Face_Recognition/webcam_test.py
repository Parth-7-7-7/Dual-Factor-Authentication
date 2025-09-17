import cv2

cap = cv2.VideoCapture(0)  # open default camera

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow('Webcam Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
