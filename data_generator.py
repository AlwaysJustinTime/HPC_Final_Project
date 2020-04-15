DATA_PER_FILE = [10,20,30,40]
TRUE_CLUSTERS_NUMBERS = [2,3,4,5]
FOLDER_NAMES = ["suite_"+str(num) for num in DATA_PER_FILE]
CLUSTER_STD = [0.25, 0.25]

from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import math

def get_unit_circle_centers(cluster_num):
    increment = (2*math.pi)/cluster_num
    answer = [(math.cos(increment*num),math.sin(increment*num)) for num in range(cluster_num)]
    return answer

centers = {cluster_num:get_unit_circle_centers(cluster_num) for cluster_num in TRUE_CLUSTERS_NUMBERS}

for cluster_num in TRUE_CLUSTERS_NUMBERS:
    plt.scatter(*zip(*centers[cluster_num]))
plt.show()

# def get_data(number_samples, centers):
#     X, y = make_blobs(n_samples=number_samples, cluster_std=CLUSTER_STD, centers=centers, n_features=2)

# plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=10, label="Cluster1")
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=10, label="Cluster2")