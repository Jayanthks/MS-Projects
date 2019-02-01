
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
UBIT = 'jayanthk'
np.random.seed(sum([ord(c) for c in UBIT]))
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
colors = ['r', 'g', 'b', 'y', 'c', 'm']
MatchThreshold = 10
y1 = [5.9, 4.6, 6.2, 4.7, 5.5, 5.0, 4.9, 6.7, 5.1, 6.0]
y2 = [3.2, 2.9, 2.8, 3.2, 4.2, 3.0, 3.1, 3.1, 3.8, 3.0]
X = np.array(list(zip(y1, y2)))
plt.scatter(f1, f2, c='black', s=7)
# Euclidean Distance Caculator

# Number of clusters
k = 3
# X  and Y coordinates of random centroids
C_x = [6.2, 6.6, 6.5]
C_y = [3.2, 3.7, 3.0]
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
plt.scatter(f1, f2, c='#050505', marker='^', s=150)
plt.scatter(C[0, 0], C[0, 1], marker='o', s=200, c=colors[0])
plt.scatter(C[1, 0], C[1, 1], marker='o', s=200, c=colors[1])
plt.scatter(C[2, 0], C[2, 1], marker='o', s=200, c=colors[2])
plt.savefig('task3_iter1_a.jpg')
C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
error = np.linalg.norm(C - C_old, None) 
colors = ['r', 'g', 'b', 'y', 'c', 'm']
c = 0        
while error != 0:
    c = c + 1
    for i in range(len(X)):
        distances = np.linalg.norm(X[i] - C, axis=1)#dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = np.copy(C)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    fig, b = plt.subplots()
    b.scatter(f1, f2, marker='^', s=150, edgecolor='black', facecolor='none')
    b.scatter(C[0, 0], C[0, 1], marker='o', s=200, c=colors[0])
    b.scatter(C[1, 0], C[1, 1], marker='o', s=200, c=colors[1])
    b.scatter(C[2, 0], C[2, 1], marker='o', s=200, c=colors[2])
    plt.savefig('task3_iter{}_b.png'.format(c))
    error = np.linalg.norm(C - C_old, None)#dist(C, C_old, None)
    
    fig, ax = plt.subplots()  
    if(c>1):
        break
    for i in range(3):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], marker='^', s=150, c=colors[i])
        ax.scatter(C[0, 0], C[0, 1], marker='o', s=200, c=colors[1])
        ax.scatter(C[1, 0], C[1, 1], marker='o', s=200, c=colors[2])
        ax.scatter(C[2, 0], C[2, 1], marker='o', s=200, c=colors[3])
    plt.savefig('task3_iter{}_a.png'.format(c+1))
    if (error == 0):
        print("")           
class KMeans():
    def __init__(self, num_clusters, tolerance=0.001, epoch=25, centroids={}):
        self.tolerance = tolerance
        self.epochs = epoch
        self.centroids = centroids
        self.num_clusters = num_clusters

    def find(self, data):
            classification = list()
            for point in data:
                distances = [np.linalg.norm(point - self.centroids[centroid]) for centroid in self.centroids]
                classification.append(distances.index(min(distances)))
            return np.array(classification)
    
    def fit(self, data):
        if self.centroids == {}:
            for i in range(self.num_clusters):
                self.centroids[i] = data[i]

        for i in range(self.epochs):
            self.classifications = {}

            for clu in range(self.num_clusters):
                self.classifications[clu] = []

            for f in data:
                distances = [np.linalg.norm(f - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))

                self.classifications[classification].append(f)
            prev_centroid = dict(self.centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            optimized = True
            for cent in self.centroids:
                original_centroid = prev_centroid[cent]
                current_centroid = self.centroids.get(cent)
                if np.sum(((original_centroid - current_centroid) * 100) * original_centroid) > self.tolerance:
                    optimized = False
            if optimized:
                break
        

def baboon(nc):
    imm = cv2.imread('D:\\sem1\\cvip\\proj2\\baboon.jpg')
    img_matrix = cv2.cvtColor(imm, cv2.COLOR_BGR2RGB)
    image = np.array(img_matrix, dtype=np.float64) / 255
    w, h, d = tuple(image.shape)
    img_arr = np.reshape(image, (w * h, d))
    image_quantization = KMeans(num_clusters=nc, epoch=30)
    image_quantization.fit(img_arr)
    lab = image_quantization.find(img_arr)
    print(image_quantization.centroids)
    img = np.zeros((w, h, 3))
    img = np.array(img, dtype=np.float64)
    label_idx = 0
    for i in range(w):
        for j in range(h):
            img[i][j] = image_quantization.centroids[lab[label_idx]]
            label_idx += 1
    matplotlib.image.imsave('task3_baboon_{}.png'.format(nc), img)



baboon(2)

