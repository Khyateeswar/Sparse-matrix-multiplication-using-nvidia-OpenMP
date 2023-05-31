module load compiler/python/3.6.0/ucs4/gnu/447
module load pythonpackages/3.6.0/numpy/1.16.1/gnu
module load suite/nvidia-hpc-sdk/21.7/cuda11.0
cd  /home/cse/btech/cs1190376/scratch/col380/A4
nvcc  -arch=sm_35 -o exec main1.cu
time nvprof ./exec input5.1 input5.2 out5

