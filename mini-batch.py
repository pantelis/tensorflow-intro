from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('mnist_cnn', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

memory_train_features_bytes = 55000 * n_input * 4 # 4 bytes for float32
memory_train_labels_bytes = 55000 * 10 * 4
memory_weights_bytes = 784 * 10 * 4
memory_bias_bytes = 10 * 4

print(memory_train_features_bytes, memory_train_labels_bytes, memory_weights_bytes, memory_bias_bytes)

