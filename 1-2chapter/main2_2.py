# 3.9 미션 실습: Shape Finder 프로젝트 🧩
# 목표:
# - 도형 이미지에서 삼각형, 사각형, 원을 자동 분류
# - 각 도형의 중심과 이름을 화면에 표시
#
# 확장 아이디어:
# - 카메라로 촬영한 객체의 형태 기반 분류기 만들기
# - EDA 프로젝트와 연결해 형태별 데이터 분석

import cv2
import numpy as np

# cap = cv2.VideoCapture(0)
img = cv2.imread('two-square.png')

while True:
    # ret, frame = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)

        cx, cy = 0, 0
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])


        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if area > 1000:
            if len(approx) == 3: shape="tri" # 삼각형
            elif len(approx) == 4: shape="quad" # 사각형
            elif len(approx) == 5: shape="penta" # 오각형
            else: shape="circle" # 원
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(img, f'{shape}, {area}', (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()