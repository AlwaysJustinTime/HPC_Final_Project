#!/bin/bash

module load cuda/9.2
module load opencv/3.4.3-contrib

clear;

cd /home/vandalovsky.v/HPC_CLASS/HPC_Final_Project/kmeans
set -o xtrace

srun --export=ALL --nodes=1 --tasks-per-node=1 --time=0:10:00 --output=results/result-kmeans-%j.out kmeans