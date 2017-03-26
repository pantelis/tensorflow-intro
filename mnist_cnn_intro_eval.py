import numpy as np
import tensorflow as tf
import os
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as plt

# Directory to put the training data.
TRAIN_DIR="./mnist_cnn"

EVAL_BATCH_SIZE = 128

# Get input data: get the sets of images and labels for training, validation, and
# test on MNIST.
data_sets = read_data_sets(TRAIN_DIR, False)

# Run evaluation based on the last saved checkpoint.
with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(
        os.path.join(TRAIN_DIR, "checkpoint-19999.meta"))
    saver.restore(
        sess, os.path.join(TRAIN_DIR, "checkpoint-19999"))


    # Retrieve the Ops we 'remembered'.
    loss = tf.get_collection("loss")
    logits = tf.get_collection("logits")
    images_placeholder = tf.get_collection("images")
    labels_placeholder = tf.get_collection("labels")
    #
    # # Add an Op that chooses the top k predictions (k=1 by default).
    # eval_op = tf.nn.top_k(logits)
    #
    # # Run evaluation.
    # images_feed, labels_feed = data_sets.validation.next_batch(EVAL_BATCH_SIZE)
    # imgplot = plt.imshow(np.reshape(images_feed, (28, 28)))
    # prediction = sess.run(eval_op,
    #                       feed_dict={images_placeholder: images_feed,
    #                                  labels_placeholder: labels_feed})
    # print("Ground truth: %d\nPrediction: %d" % (labels_feed, prediction.indices[0][0]))
