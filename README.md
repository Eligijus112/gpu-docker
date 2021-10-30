# GPU-docker

A project containing information about how to enable model training with a GPU in local machine and in docker.

# Virtual env on a local machine 

The bellow creation of a virtual environment works for both Windows and Ubuntu. The only thing that differs, is the activation.

Windows:
```
virtualenv gpu_env
source gpu_env/Scripts/activate
pip install -r requirements.txt
```

Ubuntu:
```
virtualenv gpu_env
source gpu_env/bin/activate
pip install -r requirements.txt
```

# Setting a GPU on local machine (Windows)

To set up a working Nvidia GPU on your local machine running Windows 10 refer to the steps in: https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781

The steps from the article in a nutshell: 

* Download and install cuda:

https://developer.nvidia.com/cuda-downloads 

* Download cudaNN drivers:

https://developer.nvidia.com/cudnn

* Restart your PC.

# Setting a GPU on Ubuntu

A full comprehensive guide about setting up GPU on Ubuntu can be found here: 

https://towardsdatascience.com/deep-learning-gpu-installation-on-ubuntu-18-4-9b12230a1d31 


# Checking availability of GPU

To test if the GPU is working correctly and is visible for the programs, run the following command:

```
python check_gpu.py
```

If everything is working correctly, you should see the following: 

```
$ python check_gpu.py
Is built with CUDA: True
All devices available for training: [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If there is a physical device with the GPU tag then everything is working correctly. 

# Testing out GPU on a local machine 

## Python + Tensorflow

To see the difference between CPU and GPU, run the scripts: 

```
# CPU
python train_reuters.py cpu
```
```
# GPU
python train_reuters.py
```

Running a simple DNN model to classify reuters articles the times taken were drastically different: 

For CPU script:

```
...
Epoch 48/50
9/9 [==============================] - 3s 367ms/step - loss: 0.0811 - accuracy: 0.9609 - val_loss: 1.6862 - val_accuracy: 0.8050
Epoch 49/50
9/9 [==============================] - 3s 381ms/step - loss: 0.0820 - accuracy: 0.9608 - val_loss: 1.7912 - val_accuracy: 0.7996
Epoch 50/50
9/9 [==============================] - 3s 352ms/step - loss: 0.0861 - accuracy: 0.9601 - val_loss: 1.7144 - val_accuracy: 0.8010
Training took: 165.51001071929932 seconds
```

For the GPU script: 

```
...
Epoch 48/50
9/9 [==============================] - 0s 21ms/step - loss: 0.0801 - accuracy: 0.9610 - val_loss: 1.7418 - val_accuracy: 0.8005
Epoch 49/50
9/9 [==============================] - 0s 18ms/step - loss: 0.0828 - accuracy: 0.9603 - val_loss: 1.8170 - val_accuracy: 0.7939
Epoch 50/50
9/9 [==============================] - 0s 19ms/step - loss: 0.0783 - accuracy: 0.9618 - val_loss: 1.8022 - val_accuracy: 0.7979
Training took: 10.650710821151733 seconds
```

So on a machine with a Threadripper CPU and a GTX 3090, it was faster to train the model on a GPU and not a CPU by **>16 times**. 

## R + brms

One of the popular frameworks to leverage GPU for R is OpenCL. To install OpenCL on Ubuntu run the commands: 

```
# Updating just in case
sudo apt update

# Main package
sudo apt install ocl-icd-opencl-dev
```

To test out the difference of training on a CPU vs a GPU run the code:

```
Rscript train_brms.R
```

The training time on a GPU:

```
All 4 chains finished successfully.
Mean chain execution time: 157.0 seconds.
Total execution time: 160.3 seconds
```

On a CPU: 

```
All 4 chains finished successfully.
Mean chain execution time: 692.7 seconds.
Total execution time: 705.4 seconds.
```

If a model grows in complexity so does the difference between GPU run time and CPU.

# Instructions to run the GPU in docker Windows * unstable

In a nutshell, docker has a very hard time running GPU on docker desktop. It is strongly advised to run it from an Ubuntu base. 

There are some workarounds: 

https://developer.nvidia.com/blog/announcing-cuda-on-windows-subsystem-for-linux-2/

# Reaching the GPU in docker (Ubuntu) * unstable

The bellow instructions may provide unstable results. This is because docker capabilities to access GPUs are obstructed and may require alot of tinkering.

To reach the GPU in docker, we will use an official NVIDIA docker image: https://hub.docker.com/r/nvidia/cuda/tags 

To build the image run the command:

```
docker build -t docker_gpu .
```

To run and enable the GPU in docker run:

```
docker run --gpus all docker_gpu
```

