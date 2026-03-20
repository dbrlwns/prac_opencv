import cv2
import numpy as np


print(cv2.__version__)
img = cv2.imread('two-square.png')
resized = cv2.resize(img,(500,500))
gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

# 이진화
# ret, thresh = cv2.threshold(gray,127,255,0)
# 이진화의 반전 (흰 배경 -> 검정   | 검은 모형(0) -> 흰색(255)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Contour(윤곽선) 검출, 이때 이진화된 이미지를 사용해야 함.(thresh)
contours, hierachy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)


for cnt in contours: # Contour 특징 계산
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    print(f'area : {area}, perimeter : {perimeter}')
    M = cv2.moments(cnt)

    if M['m00'] != 0:
        # 무게 중심
        cx = int(M['m10'] / M['m00']) # x좌표 가중합 / 면적(픽셀 수)
        cy = int(M['m01'] / M['m00'])
        cv2.circle(thresh,(cx,cy),5,(0,0,255),-1)



        # 울퉁불퉁한 윤곽선 둘레의 오차를 설정
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        # 다각형 근사 Polygon Approximation (Douglas-Peucker algorithm)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Rectangle"
        elif len(approx) > 6:
            shape = "Circle"
        else:
            shape = "Polygon"
        cv2.putText(thresh, shape, (cx - 30, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # convex hull (모든 꼭짓점을 감싸는 가장 작은 다각형을 구함) ,윤곽선을 감싸는 블록 다각형
    hull = cv2.convexHull(cnt)
    # cv2.drawContours(img, [hull], 0, (255, 255, 0), 2)


    # 직사각형 / 원 검출 (객체의 경계 영역을 정함), 윤곽선을 완전히 감싸는 가장 작은 직사각형 좌표
    x, y, w, h = cv2.boundingRect(cnt)
    # cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 0, 255), 2)


    # 최소 외접원 계산
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    cv2.circle(resized, (int(x), int(y)), int(radius), (255, 0, 255), 2)


# 허프 변환 Hough Transform (직섡이나 원을 수학적으로 검출)
edges = cv2.Canny(gray, 50, 150)    # gray에서 픽셀 밝기 변화가 급격한 경계선만 남김(이진 이미지)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                        minLineLength=100, maxLineGap=10) # voting으로 직선 검출
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(resized, (x1, y1), (x2, y2), (0, 0, 255), 2)


# 원 검출, 이거도 허프 변환을 사용. voting으로 검출
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
#                            param1=50, param2=30, minRadius=0, maxRadius=0)
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for c in circles[0, :]:
#         pass
#         cv2.circle(resized, (c[0], c[1]), c[2], (0, 255, 0), 2)


cv2.drawContours(img, contours, -1, (0,255,0), 2)
cv2.imshow('image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()






























