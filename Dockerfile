# Base image 
FROM nvidia/cuda:10.2-base

# Checking the availability of GPU 
CMD ["nvidia-smi"]