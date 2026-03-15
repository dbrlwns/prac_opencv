import cv2
import numpy as np

print(cv2.__version__)
img = cv2.imread('test.jpg')
resized = cv2.resize(img,(800,600))
cv2.imshow('image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()