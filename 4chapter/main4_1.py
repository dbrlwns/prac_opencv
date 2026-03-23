# 특징 매칭 Feature Matching : 두 이미지 특징점 비교하여 유사도 계산
import cv2

img1 = cv2.imread('../1-2chapter/two-square.png')
img2 = cv2.imread('building.jpg')

orb = cv2.ORB.create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], img1.copy(),  flags=2)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()