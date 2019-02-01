import cv2
import numpy as np
#reading the image
image = cv2.imread("sobel.png",0)
def flip(image):
    image_copy = np.zeros(shape=(3,3))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j] = image[image.shape[0]-i-1][image.shape[1]-j-1]
    return image_copy

def dosobel(image, kernel):
    ker = flip(kernel)
    image_h = image.shape[0]
    image_w = image.shape[1]
    ker_h = ker.shape[0]
    ker_w = ker.shape[1]
    h = ker_h//2
    w = ker_w//2
    sobel_img= np.zeros(image.shape)

    for row in range(h, image_h-h):
        for col in range(w, image_w-w):
            total = 0
            for m in range(ker_h):
                for n in range(ker_w):
                    total = (total + ker[m][n] * image[row-h+m][col-w+n])
            
            sobel_img[row][col] = total

    return sobel_img


sobel_x = np.zeros(shape=(3,3))
sobel_y = np.zeros(shape=(3,3))
# sobel kernal for x
sobel_x = np.array([
           [-1,0,1],
           [-2,0,2],
           [-1,0,1]
           ])
sobel_y = np.array([
           [1,2,1],
           [0,0,0],
           [-1,-2,-1]
           ])

sx = dosobel(image, sobel_x)
cv2.imshow("Sobel along - x", sx)
#cv2.imwrite( 'D:\sem1\cvip\hw1', sx );


sy = dosobel(image, sobel_y)
cv2.imshow("Sobel along -y", sy)

#cv2.imshow("sobel_edge", )
cv2.waitKey(0)
cv2.destroyAllWindows()

