
# coding: utf-8

# In[4]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
UBIT = 'jayanthk'
np.random.seed(sum([ord(c) for c in UBIT]))
import random

img1 = cv2.imread("D:\\sem1\\cvip\\proj2\\tsucuba_left.png", 0)
img2 = cv2.imread("D:\\sem1\\cvip\\proj2\\tsucuba_right.png", 0)
sift = cv2.xfeatures2d.SIFT_create()
(kp1, d1) = sift.detectAndCompute(img1, None)
(kp2, d2) = sift.detectAndCompute(img2, None)
sift_img1 = cv2.drawKeypoints(img1, kp1, color=(150,0,0), outImage=np.array([]))
sift_img2 = cv2.drawKeypoints(img2, kp2, color=(150,0,0), outImage=np.array([]))
print("# keypoints: {}, descriptors: {}".format(len(kp1), d1.shape))
print("# keypoints: {}, descriptors: {}".format(len(kp2), d2.shape))

cv2.imwrite("D:\\sem1\\cvip\\proj2\\task2_sift1.jpg", sift_img1)
cv2.imwrite("D:\\sem1\\cvip\\proj2\\task2_sift2.jpg", sift_img2)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(d1,d2,k=2)

# store all the good matches as per Lowe's ratio test.
g = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        g.append(m)

src_pt = np.float32([ kp1[m.queryIdx].pt for m in g ])
dst_pt = np.float32([ kp2[m.trainIdx].pt for m in g ])

F, mask = cv2.findFundamentalMat(src_pt, dst_pt, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()

draw_params = dict(matchColor = (150,0,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,g,None,**draw_params)
cv2.imwrite("D:\\sem1\\cvip\\proj2\\task2_matches_knn.jpg", img3)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(d1,d2,k=2)
print("Fundamental Matrix ")
print(F)
stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(16)
stereoMatcher.setNumDisparities(112)
stereoMatcher.setBlockSize(17)
stereoMatcher.setSpeckleRange(32)
stereoMatcher.setSpeckleWindowSize(120)
stereo = stereoMatcher.compute(img1, img2)
cv2.imwrite("D:\\sem1\\cvip\\proj2\\task2_disparity.jpg", stereo)
plt.imshow(stereo)


src_pt = np.int32(src_pt)
dst_pt = np.int32(dst_pt)

pts1 = src_pt
pts2 = dst_pt


pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

list = []
for i in range(0,11):
    y = random.randint(0,271)
    list.append(y)

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
lines11 = np.copy(lines1)
count=0
for i in list:
    lines11[count,:] = lines1[i,:]
line112 = np.copy(lines11[0:10])

img5,img6 = drawlines(img1,img2,line112,pts1,pts2)
# epilines corresponding to points in left image (first image) and drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
lines22 = np.copy(lines2)
count=0
for i in list:
    lines22[count,:] = lines2[i,:]
line122 = np.copy(lines22[0:10])
img3,img4 = drawlines(img2,img1,line122,pts2,pts1)
cv2.imwrite("D:\\sem1\\cvip\\proj2\\task2_epi1_left.jpg", img5)
cv2.imwrite("D:\\sem1\\cvip\\proj2\\task2_epi2_right.jpg", img3)

