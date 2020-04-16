DATA_PER_FILE = [10,20,30,40]
TRUE_CLUSTERS_NUMBERS = [2,3,4,5]
HEAD_FOLDER_NAME = "data"
FOLDER_NAMES = {num: "suite_"+str(num) for num in DATA_PER_FILE}
FILE_NAMES = {num: "cluster_"+str(num)+".csv" for num in TRUE_CLUSTERS_NUMBERS}
CLUSTER_STD = 0.25
GRAPH_DATA = False

from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import math
import pandas as pd
import os.path

def get_unit_circle_centers(cluster_num):
    increment = (2*math.pi)/cluster_num
    answer = [(math.cos(increment*num),math.sin(increment*num)) for num in range(cluster_num)]
    return answer

centers = {cluster_num:get_unit_circle_centers(cluster_num) for cluster_num in TRUE_CLUSTERS_NUMBERS}

def graph_centers(centers):
    for cluster_num in TRUE_CLUSTERS_NUMBERS:
        plt.scatter(*zip(*centers[cluster_num]))
    plt.show()
if GRAPH_DATA: graph_centers(centers)

def write_to_csv(data, labels, filepath):
    data = pd.DataFrame([[data[idx][0], data[idx][1], labels[idx]] for idx in range(len(data))])
    data.to_csv(filepath, index=False, header=False)

def gen_data(number_samples, centers, filepath):
    cluster_std = [CLUSTER_STD for _ in centers]
    data, labels = make_blobs(n_samples=number_samples, cluster_std=cluster_std, centers=centers, n_features=2)
    write_to_csv(data, labels, filepath)

for num_of_data in DATA_PER_FILE:
    folder_name = os.path.join(HEAD_FOLDER_NAME, FOLDER_NAMES[num_of_data])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for cluster_num in TRUE_CLUSTERS_NUMBERS:
        filepath = os.path.join(folder_name, FILE_NAMES[cluster_num])
        gen_data(num_of_data, centers[cluster_num], filepath)

# plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=10, label="Cluster1")
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=10, label="Cluster2")