# Importing the deep learning lib 
import tensorflow as tf 

# Listing the available devices 
print(f"Is built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"All devices available for training: {tf.config.list_physical_devices()}")