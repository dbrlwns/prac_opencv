import cv2
import numpy as np

# 실시간 색상 객체 추적
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # 색상이 밝기와 분리되어 있음.

    lower_blue = np.array([100, 150, 50]) # 파란색 계열, 적당히 진한, 너무 어두운 건 제외
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue) # 범위 안의 픽셀만 흰색으로, 그 외는 검정(0)으로
    res = cv2.bitwise_and(frame, frame, mask=mask) # 두 frame을 AND 연산, mask가 0인건 0(검정)으로
    # 크로마키가 이렇게 사용되는

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(res, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Tracking", res)

    if cv2.waitKey(1) == 27:  # ESC 키
        break

cap.release()
cv2.destroyAllWindows()