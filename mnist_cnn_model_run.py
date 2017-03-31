import numpy as np
import tensorflow as tf

import time
from datetime import datetime
import os.path

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import matplotlib.pyplot as plt
import mnist_cnn_intro as model

# Define all the configuration parameters as external flags. We can extend this to the parset library later

flags = tf.flags
configuration = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_integer('num_classes', 10, 'Number of classes in MNIST dataset. '
                                        'They representing the digits 0 through 9.')
flags.DEFINE_integer('image_size', 28, 'The MNIST images are always 28x28 pixels.')

flags.DEFINE_integer('batch_size', 100, 'Batch size. Must be evenly dividable by dataset sizes.')
flags.DEFINE_integer('eval_batch_size', 100, 'Evaluation batch size.')

flags.DEFINE_integer('num_channels_1', 4, 'number of output channels in conv layer 1')
flags.DEFINE_integer('num_channels_2', 8, 'number of output channels in conv layer 2')
flags.DEFINE_integer('num_channels_3', 12, 'number of output channels in conv layer 3')

flags.DEFINE_integer('filter_1_units', 5, 'conv layer 1 filter size = filter_1_units x filter_1_units')
flags.DEFINE_integer('filter_2_units', 5, 'conv layer 2 filter size = filter_2_units x filter_2_units')
flags.DEFINE_integer('filter_3_units', 4, 'conv layer 3 filter size = filter_3_units x filter_3_units')

flags.DEFINE_integer('fully_connected_units', 200, 'Fully connected layer units')

flags.DEFINE_integer('max_steps', 20000, 'Maximum number of training steps')

flags.DEFINE_string('train_dir', 'tf_logs', 'Directory to put the training data.')

configuration._parse_flags()
print('\nParameters:')
for attr, value in sorted(configuration.__flags.items()):
    print('{} = {}'.format(attr, value))
print()

# Derived Parameters
image_pixels = configuration.image_size * configuration.image_size

beginTime = time.time()

# Put logs for each run in separate directory
logdir = configuration.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

# Uncommenting these lines removes randomness
# You'll get exactly the same result on each run
# np.random.seed(1)
# tf.set_random_seed(1)

# Get input data: get the sets of images and labels for training, validation, and
# test on MNIST.
data_sets = read_data_sets(logdir, one_hot=True)

# Build the complete graph for feeding inputs, training, and saving checkpoints.
mnist_graph = tf.Graph()
with mnist_graph.as_default():
    # # Uncomment the following line to see what we have constructed.
    # tf.train.write_graph(tf.get_default_graph().as_graph_def(),
    #                      configuration.train_dir, "train_mnist_cnn.pbtxt", as_text=True)
    #
    # # Uncomment the following line to see what we have constructed.
    # tf.train.write_graph(tf.get_default_graph().as_graph_def(),
    #                      configuration.train_dir, "inference_mnist_cnn.pbtxt", as_text=True)

    # Generate placeholders for the images and labels.
    images_placeholder = tf.placeholder(tf.float32)
    labels_placeholder = tf.placeholder(tf.int64)

    tf.add_to_collection("images", images_placeholder)  # Remember this Op.
    tf.add_to_collection("labels", labels_placeholder)  # Remember this Op.

    # Build a Graph that computes predictions from the inference model.
    logits = model.inference(images_placeholder, configuration.filter_1_units, configuration.num_channels_1,
                             configuration.filter_2_units, configuration.num_channels_2,
                             configuration.filter_3_units, configuration.num_channels_3,
                             configuration.fully_connected_units, configuration.num_classes)

    tf.add_to_collection("logits", logits)  # Remember this Op.

    # Operation for the loss function
    loss = model.loss(logits, labels_placeholder)

    # Operation that calculates and applies gradients.
    train_op = model.training(loss=loss, learning_rate=configuration.learning_rate)

    # Operation calculating the accuracy of our predictions
    accuracy = model.evaluation(logits, labels_placeholder)

    # Operation merging summary data for TensorBoard
    summary = tf.summary.merge_all()

    # # Uncomment the following line to see what we have constructed.
    # tf.train.write_graph(tf.get_default_graph().as_graph_def(),
    #                      parameters.run.train_dir, "complete_mnist_cnn.pbtxt", as_text=True)

    # # finalize the graph and throw an error if modifications to the graph are done
    # tf.get_default_graph().finalize()

    # Create a saver for saving model state at checkpoints.
    saver = tf.train.Saver()

# -----------------------------------------------------------------------------
# Run the TensorFlow graph - Training
# -----------------------------------------------------------------------------
with tf.Session(graph=mnist_graph) as sess:
    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Run the Op to initialize the variables.
    sess.run(init)

    summary_writer = tf.summary.FileWriter(logdir, sess.graph)

    # Start the training loop.
    for step in range(configuration.max_steps):
        # Read a batch of images and labels.
        images_train, labels_train = data_sets.train.next_batch(configuration.batch_size)

        # reshape for CNN
        images_train = images_train.reshape(
            (configuration.batch_size, configuration.image_size, configuration.image_size, configuration.num_channels)
        )

        feed_dict = {images_placeholder: images_train, labels_placeholder: labels_train}

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss],
                                 feed_dict=feed_dict)

        # Print out loss value and the model's current accuracy
        if step % 1000 == 0:
            print('Step %d: loss = %.2f' % (step, loss_value))
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print('Step {:d}, training accuracy {:g}'.format(step, train_accuracy))
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)

        # Periodically save checkpoint
        if (step + 1) % 1000 == 0:
            checkpoint_file = os.path.join(logdir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=step)
            print('Saved checkpoint')

    endTime = time.time()
    print('Training time: {:5.2f}s'.format(endTime - beginTime))
# ---------------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------------
with tf.Session(graph=mnist_graph) as sess:

    # Start the validation loop.
    for step in range(configuration.max_steps):
        # Read a batch of images and labels.
        images_valid, labels_valid = data_sets.validation.next_batch(configuration.eval_batch_size)

        # reshape for CNN
        images_eval = images_eval.reshape(
            (configuration.batch_size, configuration.image_size, configuration.image_size, configuration.num_channels)
        )
        labels_eval = (np.arange(configuration.num_classes) == labels_eval[:, None]).astype(np.float32)

        feed_dict = {images_placeholder: images_eval, labels_placeholder: labels_eval}




# Run model validation based on the last saved checkpoint.
with tf.Session(graph=tf.Graph()) as sess:
    # The checkpoint file is just a bookkeeping file that you can use in combination of high-level helper for loading
    #  different time saved chkp files.
    # The meta chkp files hold the compressed Protobufs graph of your model and all
    #  the metadata associated (collections, learning rate, operations, etc.)
    # The chkp files holds the data (weights)
    #  itself (this one is usually quite big in size). The pbtxt file is just the non-compressed Protobufs graph of
    # your model if you want to do some debugging.
    # Finally, the events file store everything you need to visualise your model and all the data measured while you
    # were training, with TensorBoard.

    saver = tf.train.import_meta_graph(os.path.join(logdir, "checkpoint-19999.meta"))
    saver.restore(sess, os.path.join(logdir, "checkpoint-19999"))

    # Retrieve the Ops we 'remembered'.
    loss = tf.get_collection("loss")
    logits = tf.get_collection("logits")
    images_placeholder = tf.get_collection("images")
    labels_placeholder = tf.get_collection("labels")

    # Add an Op that chooses the top k predictions (k=1 by default).
    eval_op = tf.nn.top_k(logits)

    # Run evaluation.
    images_train, labels_train = data_sets.validation. .next_batch(EVAL_BATCH_SIZE)
    imgplot = plt.imshow(np.reshape(images_train, (28, 28)))
    # prediction = sess.run(eval_op,
    #                       feed_dict={images_placeholder: images_train,
    #                                  labels_placeholder: labels_train})
    # print("Ground truth: %d\nPrediction: %d" % (labels_train, prediction.indices[0][0]))

    # After finishing the training, evaluate on the test set
    # Read a batch of images and labels.
    images_train, labels_train = data_sets.train.next_batch(configuration.batch_size)
    test_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: data_sets['images_test'],
        labels_placeholder: data_sets['labels_test']})
    print('Test accuracy {:g}'.format(test_accuracy))
