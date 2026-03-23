# 이미지 분류 Image Classification
# MachineLearning Model to simple image classification using scikit-learn
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import cv2, glob

x, y = [], []
for path in glob.glob("dataset/training_set/training_set/cats/*.jpg")[:2000]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    x.append(img.flatten())
    y.append(0)

for path in glob.glob("dataset/training_set/training_set/dogs/*.jpg")[:2000]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    x.append(img.flatten())
    y.append(1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #random_state=42)
clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)

print(f"acc : {clf.score(x_train, y_train)}")

# 학습된 모델 테스트
test_img = cv2.imread("test_cat.jpg", cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(test_img, (64, 64))
test_img = test_img.flatten().reshape(1, -1) # 1장이라 reshape

result = clf.predict(test_img)
print(f'test result : {result} : {"고양이" if result[0]==0 else "강아지"}')
# SVM 방식이 이미지의 픽셀값 나열로 학습해서 (형태/패턴으로 이해하는 것이 아님) 정확도가 9짐
# 참고로 SVM은 전통적인 ML 이미지 분류
