#!/usr/bin/env python
import os

DATA_PER_FILE = [1600000]
TRUE_CLUSTERS_NUMBERS = [2,3,4,5]
HEAD_FOLDER_NAME = "data"
FOLDER_NAMES = {num: "suite_"+str(num) for num in DATA_PER_FILE}
FILE_NAMES = {num: "cluster_"+str(num)+".csv" for num in TRUE_CLUSTERS_NUMBERS}


#os.system("module load cuda/9.2")
os.system("module load opencv/3.4.3-contrib")
os.system("clear")
os.system("cd /home/vandalovsky.v/HPC_CLASS/HPC_Final_Project/kmeans")
os.system("set -o xtrace")
os.system(" rm -rf results/*.out;")


nodes = 1
tasksPerNode = 1 


# os.system("srun --export=ALL --nodes=1 --tasks-per-node=1 --time=0:10:00 --output=results/result-kmeans-%j.out kmeans 2 ../data/suite_160/cluster_2.csv")
for t in range(4):
    tasksPerNode = 2**t
    for dataSize in DATA_PER_FILE:
        for clusterNum in TRUE_CLUSTERS_NUMBERS:
            
            strCluster = str(clusterNum)
            strDataSize = str(dataSize)
            strTasks = str(tasksPerNode)
            print "launching run with data size = " + strDataSize + " & cluster # = " + strCluster + " & tasks = " + strTasks

            srun_str = "srun --export=ALL --time=0:10:00"
            srun_str += " --nodes=" + str(nodes) + " --tasks-per-node=" + strTasks
            srun_str += " --output=results/result-kmeans-" + strDataSize + "-" + strCluster + "-" + strTasks + ".out"
            srun_str += " kmeans "

            cli_args = strCluster + " ../data/suite_" + strDataSize + "_cluster_" + strCluster + ".csv"
            srun_str += cli_args
            
            os.system(srun_str)
        #end
    #end
#end
        
