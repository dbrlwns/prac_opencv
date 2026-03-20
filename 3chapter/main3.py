import cv2
import numpy as np

# cap = cv2.VideoCapture("test.mp4")
cap = cv2.VideoCapture(0)

# 얼굴 탐지 Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # edges = cv2.Canny(gray, 10, 30) # 민감
    # cv2.imshow('webcam1', edges)
    edges = cv2.Canny(gray, 100, 200) # 둔감
    cv2.imshow('webcam2', edges)


    # 얼굴 탐지 # 가볍고 빠르지만, 조명/각도에 민감함.
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()