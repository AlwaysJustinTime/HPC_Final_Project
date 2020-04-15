#!/bin/bash
cd /home/vandalovsky.v/HPC_CLASS/HPC_Final_Project/kmeans
module load cuda/9.2
module load opencv/3.4.3-contrib

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=exec_heq
#SBATCH --reservation=GPU-CLASS-SP20
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k40m:1
#SBATCH --output=exec_heq.%j.out

set -o xtrace
./heq input/bridge.png






mpicc src/main.cpp -lm -o kmean
srun --pty --export=ALL --nodes=4 --tasks-per-node=1 mpirun --mca btl_base_warn_component_unused 0 kmeans
