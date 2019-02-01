import cv2
import numpy as np 

imgc = cv2.imread("pos_14.jpg")
img_gre = cv2.imread("pos_14.jpg",0)
img = cv2.GaussianBlur(img_gre, (3,3),0.47)
img_lap = cv2.Laplacian(img_gre,cv2.CV_8U)
temp = cv2.imread("new.jpg", 0)
img_temp = cv2.Laplacian(temp,cv2.CV_8U)
h, w = img_temp.shape
res = cv2.matchTemplate(img_lap, img_temp, cv2.TM_CCOEFF_NORMED)
thres = 0.55
points = np.where(res>=thres)
for m in zip(*points[::-1]):
    x = cv2.rectangle(imgc, m, (m[0]+w, m[1]+h), (100, 10, 180), 2)

cv2.imshow('cursor detected', imgc)
cv2.imwrite("cursor_detected.png", imgc)
cv2.waitKey(0)
cv2.destroyAllWindows()