# Base image 
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# Checking the availability of GPU 
CMD ["nvvc", "-v"]