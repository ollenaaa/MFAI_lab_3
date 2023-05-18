import random
import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def kmeans(points, k):
    centers = random.sample(points, k)
    while True:
        clusters = [[] for _ in range(k)]
        for point in points:
            distances = [distance(point, center) for center in centers]
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)
        new_centers = []
        for i in range(k):
            if clusters[i]:
                new_center = tuple(map(lambda x: sum(x)/len(x), zip(*clusters[i])))
                new_centers.append(new_center)
            else:
                new_centers.append(centers[i])
        if new_centers == centers:
            break
        else:
            centers = new_centers
    return clusters, centers


def hierarchical_clustering(data, n_clusters):
    # Створюємо об'єкт ієрархічної кластеризації
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)

    # Виконуємо кластеризацію
    hierarchical_clusters = hierarchical.fit_predict(data)

    # Отримуємо мітки кластерів для кожної точки
    labels = hierarchical.labels_

    score = silhouette_score(data, hierarchical_clusters)
    # Отримуємо центри кластерів
    centers = []
    for cluster in range(n_clusters):
        cluster_points = data[labels == cluster]
        cluster_center = cluster_points.mean(axis=0)
        centers.append(cluster_center)

    # Отримуємо кількість точок у кожному кластері
    num_points = [len(data[labels == cluster]) for cluster in range(n_clusters)]

    return labels, centers, num_points, score


def calculate_cluster_quality(clusters, centers):
    sse = 0
    for i, cluster in enumerate(clusters):
        center = centers[i]
        temp = 0
        num = 0
        for point in cluster:
            temp += distance(point, center)
            num += 1
        sse += temp / num
    return sse


N = 1000
num_centers = 5
points = []

for i in range(N):
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    points.append((x, y))

kmeans_clusters, kmeans_centers = kmeans(points, num_centers)

points = np.array(points)

hierarchical_clusters, hierarchical_centers, hierarchical_num, hierarchical_score = hierarchical_clustering(points, num_centers)

kmeans_quality = calculate_cluster_quality(kmeans_clusters, kmeans_centers)

for i, cluster in enumerate(kmeans_clusters):
    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]
    plt.scatter(x, y)

for center in kmeans_centers:
    plt.scatter(center[0], center[1], s=100, color='brown', marker='x')
plt.title("K-Means Clustering")
plt.show()

plt.scatter(points[:, 0], points[:, 1], c=hierarchical_clusters, cmap='viridis')
for center in hierarchical_centers:
    plt.scatter(center[0], center[1], s=100, color='brown', marker='x')
plt.title("Hierarchical Clustering")
plt.show()

print("K-means:")
print("Number of clusters:", num_centers)
for i in range(num_centers):
    print("Center ", i + 1, " = ", kmeans_centers[i])
    print("Number of points = ", len(kmeans_clusters[i]))
print("Cluster sizes:", kmeans_quality)

print("\nHierarchical Clustering:")
print("Number of clusters:", num_centers)
for i in range(num_centers):
    print("Center ", i + 1, " = ", hierarchical_centers[i])
    print("Number of points = ", hierarchical_num[i])
print("Cluster sizes:", hierarchical_score)
