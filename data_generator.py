DATA_PER_FILE = [20,40,80,160]
TRUE_CLUSTERS_NUMBERS = [2,3,4,5]
HEAD_FOLDER_NAME = "data"
DATA_FILENAMES = {num: "suite_"+str(num) for num in DATA_PER_FILE}
CLUSTER_FILENAMES = {num: "cluster_"+str(num)+".csv" for num in TRUE_CLUSTERS_NUMBERS}
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

def graph_data(data, labels, centers):
    plt.scatter(*zip(*centers), color="red", s=20, label="Centers")
    for i in range(len(centers)):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], s=10, label="Cluster"+str(i))
    plt.axis([-2, 2, -2, 2])
    plt.show()

def write_to_csv(data, labels, filepath):
    data = pd.DataFrame([[data[idx][0], data[idx][1], labels[idx]] for idx in range(len(data))])
    data.to_csv(filepath, index=False, header=False)

def gen_data(number_samples, centers, filepath):
    cluster_std = [CLUSTER_STD for _ in centers]
    data, labels = make_blobs(n_samples=number_samples, cluster_std=cluster_std, centers=centers, n_features=2)
    write_to_csv(data, labels, filepath)
    if GRAPH_DATA: graph_data(data, labels, centers)

centers = {cluster_num:get_unit_circle_centers(cluster_num) for cluster_num in TRUE_CLUSTERS_NUMBERS}

for num_of_data in DATA_PER_FILE:
    if not os.path.exists(HEAD_FOLDER_NAME):
        os.makedirs(HEAD_FOLDER_NAME)
    for cluster_num in TRUE_CLUSTERS_NUMBERS:
        filename = DATA_FILENAMES[num_of_data] + "_" + CLUSTER_FILENAMES[cluster_num]
        filepath = os.path.join(HEAD_FOLDER_NAME, filename)
        gen_data(num_of_data, centers[cluster_num], filepath)