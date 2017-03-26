# Solution is available in the other "quiz_solution.py" tab
import tensorflow as tf

def get_weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    #  Return weights
    weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
    return weights
    pass


def get_biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    #  Return biases
    bias =  tf.Variable(tf.zeros(n_labels))
    return bias
    pass


def linear(input, w, b):
    """
    Return linear function in TensorFlow
    :param input: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """
    # Linear Function (xW + b)
    y = tf.add(tf.matmul(input,w), b)
    return y
    pass