import math
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# Define the inference operation.
def inference(images, filter_1_units, num_channels_1, filter_2_units, num_channels_2, filter_3_units, num_channels_3,
              hidden_4_units, softmax_units):
    """Build the model up to where it may be used for inference.
    Args:
        images: Images placeholder.
        parameters.graph.filter_1_units: Size of the first conv layer.
        parameters.graph.filter_2_units: Size of the second conv layer.
        parameters.graph.filter_3_units: Size of the third conv layer.
        hidden_4_units: Size of the fourth hidden linear layer.
        softmax_units: Size of the output softmax layer.
    Returns:
        logits: Output tensor with the computed logits.
    """

    # Conv Layer 1
    with tf.name_scope('layer1'):
        W1 = tf.Variable(
            tf.truncated_normal([filter_1_units, filter_1_units, 1, num_channels_1],
                                stddev=1.0 / math.sqrt(float(filter_1_units**2))), name='W1')
        B1 = tf.Variable(tf.zeros([num_channels_1]), name='B1')
        Y1 = tf.nn.relu(tf.nn.conv2d(images, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
    # Conv Layer 2
    with tf.name_scope('layer2'):
        W2 = tf.Variable(
            tf.truncated_normal([filter_2_units, filter_2_units, num_channels_1, num_channels_2],
                                stddev=1.0 / math.sqrt(float(filter_2_units^2))), name='W2')
        B2 = tf.Variable(tf.zeros([num_channels_2]), name='B2')
        Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 2, 2, 1], padding='SAME') + B2)
    # Layer 3
    with tf.name_scope('layer3'):
        W3 = tf.Variable(
            tf.truncated_normal([filter_3_units, filter_3_units, num_channels_2, num_channels_3],
                                stddev=1.0 / math.sqrt(float(filter_3_units**2))), name='W3')
        B3 = tf.Variable(tf.zeros([num_channels_3]), name='B3')
        Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 2, 2, 1], padding='SAME') + B3)
        
        # flatten all values for the subsequent fully connected layer
        Y3 = tf.reshape(Y3, shape = [-1, 7*7*num_channels_3])
    
    # Fully connected Linear
    with tf.name_scope('linear_layer4'):
        W4 = tf.Variable(
            tf.truncated_normal([7*7*num_channels_3, hidden_4_units],
                                stddev=1.0 / math.sqrt(float(7*7*num_channels_3))), name='W4')
        B4 = tf.Variable(tf.zeros([hidden_4_units]), name='B4')
        Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
    
    # Softmax Output Layer
    with tf.name_scope('softmax'):
        W5 = tf.Variable(
            tf.truncated_normal([hidden_4_units, softmax_units],
                                stddev=1.0 / math.sqrt(float(hidden_4_units)), name='W5'))
        B5 = tf.Variable(tf.zeros([softmax_units]), name='B5')
        logits = tf.matmul(Y4, W5) + B5

    return logits

# define the loss operation
def loss(logits, labels):
    """Calculates the loss from logits and labels.

    Args:
        logits: Logits tensor, float - [batch_size, parameters.dataset.num_classes].
        labels: Labels tensor, int32 - [batch_size], with values in the
          range [0, num_classes).
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        loss: The Op for calculating loss.
    """
    # Create an operation that calculates loss.
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)

    return loss

# Training operation
def training(loss, learning_rate):
    """Setup the training operation.

    Args:
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
    """

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

# define the accuracy operation
def evaluation(logits, labels):
    '''Evaluates the quality of the logits at predicting the label.

      Args:
        logits: Logits tensor, float - [batch size, number of classes].
        labels: Labels tensor, int64 - [batch size].

      Returns:
        accuracy: the percentage of images where the class was correctly predicted.
      '''
    
    with tf.name_scope('Accuracy'):
        # Operation comparing prediction with true label
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)

        # Operation calculating the accuracy of the predictions
        accuracy = 100.0 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Summary operation for the accuracy
        tf.summary.scalar('accuracy', accuracy)

    return accuracy
