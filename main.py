import cv2
import numpy as np
import matplotlib.pyplot as plt
print(f'cv2 version : {cv2.__version__}')

img = cv2.imread('test.jpg')
resized = cv2.resize(img, (800,600))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# 이진화 Thresholding
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# 적응형 이진화
adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)
# 블러링 Blurring
blur = cv2.blur(gray, (5, 5))
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
# 양방향 블러 : 가장 자연스럽게 노이즈를 줄이고 경계선 유지
bilateral = cv2.bilateralFilter(gray, 5, 75, 75)

# 샤프닝 Sharpening
kernel = np.array([[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]])
sharp = cv2.filter2D(gray, -1, kernel)

# 엣지 검출 Edge Detection - Sobel, Canny algorithm
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
edge_sobel = cv2.magnitude(sobelx, sobely)

edges = cv2.Canny(gray, 100, 200) # 노이즈 제거 -> gradient 계산 -> 임계값 과정

# 형태학적 연산 Morphology ???
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

erosion = cv2.erode(gray, kernel, iterations=1) # 침식 (노이즈 제거
dilation = cv2.dilate(gray, kernel, iterations=1) # 팽창 (끊어진 경계 복원
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel) # 열기 (작은 점 제거
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel) # 닫기 (구멍 채우기


# 히스토그램가 명암 조절 (밝기분포 분석, 대비 향상
plt.hist(gray.ravel(), 256, (0, 256))
plt.title("Histogram")
plt.show()

equalized = cv2.equalizeHist(gray) # 히스토그램 평활화
cv2.imshow('equalized', equalized)

cv2.waitKey(0)
cv2.destroyAllWindows()


# 그레이스케일 - 명도 기반 처리
# 이진화 - 밝기 기준 분리
# 블러링 - 노이즈 감소
# 샤프닝 - 엣지 강조
# 에지 검출 - 윤곽선 탐지
# 형태학 연산 - 구조 정제
# 히스토그램 평활화 - 대비 향상
