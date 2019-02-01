import cv2
import numpy as np

#image read as grey scale and color.
grey_image = cv2.imread("task2.jpg", 0)
color_image = cv2.imread("task2.jpg")

#given sigma list
sigmalt = np.zeros(shape=(4, 5))
sigmalt[0, 0] = (1 / (2 ** 0.5))
sigmalt[0, 1] = 1
sigmalt[0, 2] = 2 ** 0.5
sigmalt[0, 3] = 2
sigmalt[0, 4] = 2 * (2 ** 0.5)
sigmalt[1, 0] = 2 ** 0.5
sigmalt[1, 1] = 2
sigmalt[1, 2] = 2 * (2 ** 0.5)
sigmalt[1, 3] = 4
sigmalt[1, 4] = 4 * (2 ** 0.5)
sigmalt[2, 0] = 2 * (2 ** 0.5)
sigmalt[2, 1] = 4
sigmalt[2, 2] = 4 * (2 ** 0.5)
sigmalt[2, 3] = 8
sigmalt[2, 4] = 8 * (2 ** 0.5)
sigmalt[3, 0] = 4 * (2 ** 0.5)
sigmalt[3, 1] = 8
sigmalt[3, 2] = 8 * (2 ** 0.5)
sigmalt[3, 3] = 16
sigmalt[3, 4] = 16 * (2 ** 0.5)

keypoints = 0

def get_kp(up_img,middle,low_img,keypoints):
	 
    upper = up_img
    lower = low_img
    total_kp = keypoints
    row = middle.shape[0]
    col = middle.shape[1]
    for m in range(1,row-1):
        for n in range(1,col-1):
            cmin= 0
            cmax = 0
            if ( middle[m][n] > max( middle[m-1][n-1], middle[m][n-1], middle[m+1][n-1],middle[m-1][n],middle[m+1][n],middle[m-1][n+1],middle[m][n+1],middle[m+1][n+1]  )     ):
                cmax = cmax+1
            if ( middle[m][n] > max( upper[m-1][n-1], upper[m][n-1], upper[m+1][n-1], upper[m-1][n], upper[m][n], upper[m+1][n], upper[m-1][n+1], upper[m][n+1], upper[m+1][n+1]  )     ):
                cmax = cmax+1
            if ( middle[m][n] > max( lower[m-1][n-1], lower[m][n-1], lower[m+1][n-1], lower[m-1][n], lower[m][n], lower[m+1][n], lower[m-1][n+1], lower[m][n+1], lower[m+1][n+1]  )     ):
                cmax = cmax+1
            if ( middle[m][n] < min( middle[m-1][n-1], middle[m][n-1], middle[m+1][n-1], middle[m-1][n], middle[m+1][n], middle[m-1][n+1], middle[m][n+1], middle[m+1][n+1]  )     ):
                cmin = cmin+1
            if ( middle[m][n] < min( upper[m-1][n-1], upper[m][n-1], upper[m+1][n-1], upper[m-1][n], upper[m][n], upper[m+1][n], upper[m-1][n+1], upper[m][n+1], upper[m+1][n+1]  )     ):
                cmin = cmin+1
            if ( middle[m][n] < min( lower[m-1][n-1], lower[m][n-1], lower[m+1][n-1], lower[m-1][n], lower[m][n], lower[m+1][n], lower[m-1][n+1], lower[m][n+1], lower[m+1][n+1]  )     ):
                cmin = cmin+1
            if ( cmin==0 and cmax==3):
                total_kp = total_kp + 1
                print(total_kp," kepoint found at :  ",m," ",n, middle[m][n],"\n")
            elif (cmin == 3 and cmax == 0):
                total_kp = total_kp + 1
                print(total_kp," keypoint found at :" , m , " ",n ,middle[m][n], "\n")
                
def flip(img):
    img_c = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_c[i][j] = img[img.shape[0] - i - 1][img.shape[1] - j - 1]
    return img_c

def do_conv(img, ker):
    kernel = flip(ker)
    ker_h = kernel.shape[0]
    ker_w = kernel.shape[1]
    h = img.shape[0]
    w = img.shape[1]
    height = ker_h // 2
    width = ker_w // 2
    img_conv = np.zeros(img.shape)
    for e in range(height, h - height):
        for f in range(width, w - width):
            total = 0

            for m in range(ker_h):
                for n in range(ker_w):
                    total = total + kernel[m][n] * img[e - h + m][f - w + n]

            img_conv[e][f] = total

    return img_conv



def kernel_gauss(len, s):
    ax = np.arange(-len // 2 + 1., len // 2 + 1.)
    x, y = np.meshgrid(ax, ax)

    ker = np.exp(-(x ** 2 + y ** 2) / (2. * s ** 2))

    return ker / np.sum(ker)



def fun_resize(image):
    h, w = image.shape
    f = 2
    f_h = int(h//2)
    f_w = int(w//2)

    newimg = np.zeros((f_h, f_w), dtype = int)

    row = 0
    col = 0

    r= 0
    c= 0
    while(row<h-1):
        c = 0
        col = 0
        while(col<w-1):
            newimg[r][c] = image[row][col]
            col = col + 2
            c = c + 1
        r = r + 1
        row = row + 2
    return newimg


img_resize1= fun_resize(grey_image)
img_resize2 = fun_resize(img_resize1)
img_resize3 = fun_resize(img_resize2)
r_c = 1
y = 1
d_c = 1
#dog201 , dog202,dog203,dog204 = 0
#dog301 , dog302, dog303,dog304 =0
for row in sigmalt:
    s_c = 0
    for sigma in row:

        if (r_c == 1):

            g = kernel_gauss(7, sigma)

            sub_ker = do_conv(grey_image, g)
            name = 'Gauss/GaussianImage' + str(y) + '.png'

            cv2.imwrite(name, sub_ker)
            if (s_c >= 1):
                g = k2 - sub_ker
                dog = 'DoG/DifferenceOfGaussian' + str(d_c) + '.png'
                d_c = d_c + 1
                cv2.imwrite(dog, g)

            k2 = sub_ker
            y = y + 1
            s_c = s_c + 1

        elif (r_c == 2):

            g = kernel_gauss(7, sigma)

            h = int(grey_image.shape[0])
            w = int(grey_image.shape[1])

            s_1 = img_resize1

            sub_ker = do_conv(s_1, g)
            name = 'Gauss/GaussianImage' + str(y) + '.png'

            cv2.imwrite(name, sub_ker)
            if (s_c >= 1):
                g = k2 - sub_ker
                dog = 'DoG/DifferenceofGaussian' + str(d_c) + '.png'
                d_c = d_c + 1
                cv2.imwrite(dog, g)
                cv2.imwrite(dog, g)
                if (d_c == 6):
                    global dog201, dog202, dog203, dog204
                    dog201 = g
                if (d_c == 7):
                    dog202 = g
                if (d_c == 8):
                    dog203 = g
                if (d_c == 9):
                    dog204 = g
                    

            k2 = sub_ker
            cv2.imwrite(name, sub_ker)
            y = y + 1
            s_c = s_c + 1

        elif (r_c == 3):

            g = kernel_gauss(7, sigma)

            h = int(grey_image.shape[0])
            w = int(grey_image.shape[1])

            s_2 = img_resize2

            sub_ker = do_conv(s_2, g)
            name = 'Gauss/GaussianImage' + str(y) + '.png'

            cv2.imwrite(name, sub_ker)
            if (s_c >= 1):
                g = k2 - sub_ker
                dog = 'DoG/DifferenceOfGaussian' + str(d_c) + '.png'
                d_c = d_c + 1
                cv2.imwrite(dog, g)
                if (d_c == 10):
                    global dog301, dog302, dog303, dog304
                    dog301 = g
                if (d_c == 11):
                    # global dog222
                    dog302 = g
                if (d_c == 12):
                    # global dog333
                    dog303 = g
                if (d_c == 13):
                    # global dog444
                    dog304 = g

            k2 = sub_ker
            y = y + 1
            s_c = s_c + 1

        else:            
#cv2.imwrite(name, sub_ker)

            g = kernel_gauss(7, sigma)

            h = int(grey_image.shape[0])
            w = int(grey_image.shape[1])

            s_3 = img_resize3
            sub_ker = do_conv(s_3, g)
            name = 'GaussianImage' + str(y) + '.png'
            cv2.imwrite(name, sub_ker)
            if (s_c >= 1):
                g = k2 - sub_ker
                dog = 'DifferenceOfGaussian' + str(d_c) + '.png'
                d_c = d_c + 1
                cv2.imwrite(dog, g)
            k2 = sub_ker
            cv2.imwrite(name, sub_ker)
            y = y + 1
            s_c = s_c + 1

    r_c = r_c + 1
	
print("Keypoints of octave 2 - part1:")
get_kp(dog201,dog202,dog203,keypoints)
print("Keypoints of octave 2 - part2:")
get_kp(dog202,dog203,dog204,keypoints)
print("Keypoints of octave 3 - part1:")
get_kp(dog301,dog302,dog303,keypoints)
print("Keypoints of octave 3 - part2:")
get_kp(dog302,dog303,dog304,keypoints)







