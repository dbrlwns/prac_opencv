"""
openCV의 활용
1. 전통적 특징 기반 - SIFT, ORB, SURF : 객체 매칭, 영상 정합
2. 통계 기반 학습 (생략) - 분류기
3. 딥러닝 기반 - DNN, YOLO, SSD : 실시간 객체 인식
"""
import cv2
# 특징 : 이미지에서 다른 부분과 구별되는 시각적 패턴

# SIFT Scale-Invariant Feature Transform

img = cv2.imread('../1-2chapter/two-square.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT.create()
# keypoints는 특징점들의 위치/방향/크기, descriptors는 특징점 주변 패턴을 128차원 숫자 배열로 표현
keypoints, descriptors = sift.detectAndCompute(gray, None)

img_sift = cv2.drawKeypoints(img, keypoints, img.copy(), # copy대신 None 해도됨.
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# ORB Oriented FAST and Rotated BRIFF
# SIFT보다 빠르고, 특허 제한이 없음, 로봇-임베디드에서 주로 사용
orb = cv2.ORB.create()
kp, des = orb.detectAndCompute(gray, None)

out = cv2.drawKeypoints(img, kp, img.copy(), color=(0, 255, 0))





# print(descriptors.shape)
# cv2.imshow('img', img_sift)
cv2.imshow('img', out)
cv2.waitKey(0)
cv2.destroyAllWindows()