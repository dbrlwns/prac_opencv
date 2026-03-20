"""
카메라 필터
- 얼굴 탐지 + 색상 필터 사용
- 객체에 반투명 마스트 적용

"""

import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    overlay = frame.copy()  # 현재(원본) 프레임을 복사, frame에는 노란 사각형을 만들기 때문
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), -1)

    alpha = 0.7
    # 두 이미지를 섞음, overlay 30%, frame 70% ==> 노랑이 반투명 노랑이 됨
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
