#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import numpy as np
org_img = cv2.imread('D:\\sem1\\cvip\\proj3\\original_imgs\\noise.jpg',0)
img = org_img.copy()
img2 = org_img.copy()
img_len=img.shape
r = img_len[0]
c = img_len[1]
print(img[250][150])
#erosion
def erosion(img):
    w = img.shape[0]
    h = img.shape[1]
    dil = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            if(img[i][j]==255):
                if(i==0 or j ==0 or i==w-1 or j==h-1):
                    continue
            if( img[i-1][j-1] ==255 and img[i-1][j] ==255 and img[i-1][j+1]==255 and img[i][j-1] ==255 and img[i][j]==255 and img[i][j+1]==255 and img[i+1][j-1]==255 and img[i+1][j]==255 and img[i+1][j+1]==255):
                dil[i][j] = 255
            
    return dil 
#dilation     
def dilation(img_r,r,c):
    for i in range(0,r-2):
        for j in range(0,c-2):
            if( img_r[i][j] ==255 or img_r[i][j+1] ==255 or img_r[i][j+2]==255 or img_r[i+1][j] ==255 or img_r[i+1][j+2]==255 or img_r[i+2][j]==255 or img_r[i+2][j+1]==255 or img_r[i+2][j+2]==255):
                img_r[i][j] = 255
    return img_r

# algorithm 1 - opening followed by closing
# opening - erosion followed by dilation
img2 = erosion(img)
img4 = img2.copy()
img3 = dilation(img4,r,c)

#closing - dilation followed by erosion
img5 = dilation(img3,r,c)
img6 = img5.copy()
img7 = erosion(img6)

#algorithm 2- clsoing followed by opening
# closing - erosion followed by dilation
imgxx = dilation(img,r,c)
imgx1 = imgxx.copy()
imgyy = erosion(imgx1)

#opening - dilation followed by erosion
imgy1 = erosion(imgyy)
imgy2 = imgy1.copy()
imgy3 = dilation(imgy2,r,c)

     
     
cv2.imwrite("D:\\sem1\\cvip\\proj3\\original_imgs\\res_noise1.jpg",img7)
cv2.imwrite("D:\\sem1\\cvip\\proj3\\original_imgs\\res_noise2.jpg",imgy3)


# In[7]:


# check if both the obtained images are equal
print(np.array_equal(img3,opening))


# In[12]:


# boundary extraction :
#   1) do erosion of res_noise1 and subtract it with res_noise1(original image)
#   2) do erosion of res_noise2 and subtract it with res_noise2(original image)

b_img1 = img7.copy()
b_img2 = imgy3.copy()
b_img11 = erosion(b_img1)
b_img22 = erosion(b_img2)

res_bound1 = b_img1 - b_img11
res_bound2 = b_img2 - b_img22    
    
cv2.imwrite("D:\\sem1\\cvip\\proj3\\original_imgs\\res_bound1.jpg",res_bound1)
cv2.imwrite("D:\\sem1\\cvip\\proj3\\original_imgs\\res_bound2.jpg",res_bound2)


# In[ ]:




