
# coding: utf-8

# In[24]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
UBIT = 'jayanthk'
np.random.seed(sum([ord(c) for c in UBIT]))
import random
# Reading the images in grey scale
img1 = cv2.imread("D:\\sem1\\cvip\\proj2\\mountain1.jpg", 0)
img2 = cv2.imread("D:\\sem1\\cvip\\proj2\\mountain2.jpg", 0)
img1_color = cv2.imread("D:\\sem1\\cvip\\proj2\\mountain1.jpg", 1)
img2_color= cv2.imread("D:\\sem1\\cvip\\proj2\\mountain2.jpg", 1)
# calling the SIFT fucntion
sift = cv2.xfeatures2d.SIFT_create()
#computing the sift features for img1 and img2
(kp1, d1) = sift.detectAndCompute(img1, None)
(kp2, d2) = sift.detectAndCompute(img2, None)
#drawing keypoints
sift1 = cv2.drawKeypoints(img1, kp1, color=(150, 0, 0), outImage=np.array([]))
sift2 = cv2.drawKeypoints(img2, kp2, color=(150, 0, 0), outImage=np.array([]))
match_c = 10
print("# keypoints: {}, descriptors: {}".format(len(kp1), d1.shape))
print("# keypoints: {}, descriptors: {}".format(len(kp2), d2.shape))
cv2.imwrite("task1 sift1.jpg", sift1)
cv2.imwrite("task1 sift2.jpg", sift2)
FLANN_INDEX_KDTREE = 0
#index parameters and search parameters with FLANN
indexp= dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchp = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexp, searchp)
matches = flann.knnMatch(d1, d2, k=2)
g = []
#knn matcher with the condition mentioned
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        g.append(m)
# Fucntion to find inliners - random selection
if (len(g) > match_c):
    #source points 
    src_pt = np.float32([kp1[m.queryIdx].pt for m in g]).reshape(-1, 1, 2)
    # destiantion points
    dst_pt = np.float32([kp2[m.trainIdx].pt for m in g]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pt, dst_pt, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    Mask1 = (mask.ravel()==1).tolist()
    #list to the random points
    list_t = []
    #randomly generated points for selecting the inliers
    for i in range(0, 10):
        x = random.randint(0, 244)
        list_t.append(x)
    for i in range(0,len(Mask1)):
        if (i in list_t):
            y =0
        else:
            Mask1[i]=False
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
#function for wrapping the image - mostly made up of fucntion call
def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1
    return output_img

warp = warpImages(img2_color, img1_color, H)
draw_params = dict(matchColor=(0, 0, 240),  
                   singlePointColor=None,
                   matchesMask=matchesMask,  
                   flags=2)
params1 = dict(matchColor=(0, 100, 0),  
                   singlePointColor=None,
                   matchesMask=Mask1, 
                   flags=2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, g, None, **draw_params)

cv2.imwrite("task1_pano.jpg", warp)
cv2.imwrite("task1_matches_knn.jpg", img3)
img4 = cv2.drawMatches(img1_color, kp1, img2_color, kp2, g, None, **params1)
cv2.imwrite("task1_matches.jpg", img4)
print("The homography matrix is")
print(H)

