# Step 1: Import libraries.
import math
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


# Step 2: Define some constants.
# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Batch size. Must be evenly dividable by dataset sizes.
BATCH_SIZE = 100
EVAL_BATCH_SIZE = 1

# CNN - number of output channels per layer
num_channels_1 = 4
num_channels_2 = 8
num_channels_3 = 12

# Convolutional Layers- filter sizes
filter_1_units = 5 # 5x5
filter_2_units = 5 # 5x5
filter_3_units = 4 # 4x4

# FULLY CONNECTED LAYER 
fully_connected_units = 200

# SOFTMAX OUTPUT LAYER
# NUM_CLASSES = 10 - defined above

# Maximum number of training steps.
MAX_STEPS = 20000

# Directory to put the training data.
TRAIN_DIR="/home/pantelis/Projects/tensorflow-intro/mnist_cnn"


# Step 3: Get input data: get the sets of images and labels for training, validation, and
# test on MNIST.
data_sets = read_data_sets(TRAIN_DIR, one_hot=True)


# Step 4: Build inference graph.
def mnist_inference(images, filter_1_units, num_channels_1, filter_2_units, num_channels_2, filter_3_units, num_channels_3, 
                    hidden_4_units, softmax_units):
    """Build the MNIST model up to where it may be used for inference.
    Args:
        images: Images placeholder.
        filter_1_units: Size of the first conv layer.
        filter_2_units: Size of the second conv layer.
        filter_3_units: Size of the third conv layer.
        hidden_4_units: Size of the fourth hidden linear layer.
        softmax_units: Size of the output softmax layer.
    Returns:
        logits: Output tensor with the computed logits.
    """
    # Conv Layer 1
    with tf.name_scope('layer1'):
        W1 = tf.Variable(
            tf.truncated_normal([filter_1_units, filter_1_units, 1, num_channels_1],
                                stddev=1.0 / math.sqrt(float(filter_1_units^2))), name='W1')
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
                                stddev=1.0 / math.sqrt(float(filter_3_units^2))), name='W3')
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

    # Uncomment the following line to see what we have constructed.
    tf.train.write_graph(tf.get_default_graph().as_graph_def(),
                          "TRAIN_DIR", "inference_mnist_cnn.pbtxt", as_text=True)
    return logits

# define the loss operation
def loss(logits, labels):
    """Calculates the loss from logits and labels.

    Args:
        logits: Logits tensor, float - [BATCH_SIZE, NUM_CLASSES].
        labels: Labels tensor, int32 - [BATCH_SIZE], with values in the
          range [0, NUM_CLASSES).
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        loss: The Op for calculating loss.
    """
    # Create an operation that calculates loss.
    labels = tf.to_int64(labels)
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    return loss

# Step 5: Build training graph.
def mnist_training(loss, learning_rate):
    """Build the training graph.

    Args:
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Uncomment the following line to see what we have constructed.
    tf.train.write_graph(tf.get_default_graph().as_graph_def(),
                         TRAIN_DIR, "train_mnist_cnn.pbtxt", as_text=True)
    
    return train_op

# define the accuracy operation
def evaluation(logits, labels):
    with tf.name_scope('Accuracy'):
        # Operation comparing prediction with true label
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)

        # Operation calculating the accuracy of the predictions
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Summary operation for the accuracy
        tf.scalar_summary('train_accuracy', accuracy)

    return accuracy

# Step 6: Build the complete graph for feeding inputs, training, and saving checkpoints.
mnist_graph = tf.Graph()
with mnist_graph.as_default():
    # Generate placeholders for the images and labels.
    images_placeholder = tf.placeholder(tf.float32)                                       
    labels_placeholder = tf.placeholder(tf.int32)
    
    tf.add_to_collection("images", images_placeholder)  # Remember this Op.
    tf.add_to_collection("labels", labels_placeholder)  # Remember this Op.

    # Build a Graph that computes predictions from the inference model.
    logits = mnist_inference(images_placeholder,filter_1_units, num_channels_1, filter_2_units, num_channels_2, 
                             filter_3_units, num_channels_3, fully_connected_units, NUM_CLASSES)
    tf.add_to_collection("logits", logits)  # Remember this Op.

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist_training(logits=logits, labels=labels_placeholder, learning_rate=0.01)

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    
    # Uncomment the following line to see what we have constructed.
    tf.train.write_graph(tf.get_default_graph().as_graph_def(),
                         TRAIN_DIR, "complete_mnist_cnn.pbtxt", as_text=True)

    # finalize the graph and throw an error if modifications to the graph are done
    tf.get_default_graph().finalize()

# Step 7: Run training for MAX_STEPS and save checkpoint at the end.
with tf.Session(graph=mnist_graph) as sess:
    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in range(MAX_STEPS):
        # Read a batch of images and labels.
        images_feed, labels_feed = data_sets.train.next_batch(BATCH_SIZE)

        images_feed = images_feed.reshape((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss],
                                 feed_dict={images_placeholder: images_feed,
                                            labels_placeholder: labels_feed})

        # Print out loss value.
        if step % 1000 == 0:
            print('Step %d: loss = %.2f' % (step, loss_value))

        # Write a checkpoint.
        checkpoint_file = os.path.join(TRAIN_DIR, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=step)

    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()