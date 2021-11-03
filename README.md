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

## Python + Tensorflow (Windows)

System specs: 

* Windows 10 Pro 
* 256 GB RAM 
* AMD Threadripper 3990X
* Geforce RTX 3090

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

It was faster to train the model on a GPU and not a CPU by **>16 times**. 

## Python + Tensorflow (Ubuntu)

System specs: 

* Ubuntu 20.04 LTS
* 64 GB RAM
* AMD Ryzen 7 5800X 
* Geforce RTX 3080


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
9/9 [==============================] - 1s 113ms/step - loss: 0.0827 - accuracy: 0.9596 - val_loss: 1.7463 - val_accuracy: 0.8005
Epoch 49/50
9/9 [==============================] - 1s 112ms/step - loss: 0.0847 - accuracy: 0.9603 - val_loss: 1.7179 - val_accuracy: 0.8005
Epoch 50/50
9/9 [==============================] - 1s 112ms/step - loss: 0.0825 - accuracy: 0.9618 - val_loss: 1.7377 - val_accuracy: 0.8010
Training took: 51.729074239730835 seconds
```

For the GPU script: 

```
...
Epoch 48/50
9/9 [==============================] - 0s 16ms/step - loss: 0.0846 - accuracy: 0.9621 - val_loss: 1.6438 - val_accuracy: 0.8050
Epoch 49/50
9/9 [==============================] - 0s 15ms/step - loss: 0.0815 - accuracy: 0.9631 - val_loss: 1.8274 - val_accuracy: 0.8028
Epoch 50/50
9/9 [==============================] - 0s 15ms/step - loss: 0.0864 - accuracy: 0.9580 - val_loss: 1.7148 - val_accuracy: 0.8028
Training took: 8.480462789535522 seconds
```

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

The official guide by Nvidia: 

```
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
```

After following the installation you can check if everything is working correctly by using the command: 

```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

The results should look similar to: 

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.42.01    Driver Version: 470.42.01    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:09:00.0  On |                  N/A |
|  0%   34C    P8    15W / 370W |    871MiB /  9995MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+

```

If one can see the above table then, in theory, applications leveraging GPU should work in docker. 

## Python + docker + gpu 

Building the image: 

```
docker build -t python_gpu_docker -f Dockerfile-py .
```

Running the model training on a GPU in docker:

```
docker run --gpus all --runtime nvidia python_gpu_docker
```

## R + docker + gpu 

Building the image: 

```
docker build -t r_gpu_docker -f Dockerfile-R .
```

Running the file: 

```
docker run --gpus all --runtime nvidia r_gpu_docker 
```

