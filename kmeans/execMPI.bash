#!/bin/bash
cd /home/vandalovsky.v/HPC_CLASS/HPC_Final_Project/kmeans
module load cuda/9.2
module load opencv/3.4.3-contrib
srun --pty --export=ALL --nodes=4 --tasks-per-node=1 mpirun --mca btl_base_warn_component_unused 0 kmeans_mpi
