# Solution is available in the other "solution.py" tab
import numpy as np
import tensorflow as tf

def softmax(x):
    '''
    >>> res = softmax(np.array([0, 200, 10]))
    >>> np.sum(res)
    1.0
    >>> np.all(np.abs(res - np.array([0, 1, 0])) < 0.0001)
    True
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200], [200, 0, 10]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.,  1.])
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.])
    '''
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

logits = np.array([
    [1, 2, 3, 6],
    [2, 4, 5, 6],
    [3, 8, 7, 6]])

s = softmax(logits)
print(sum(s.transpose()))
print(s)

output = None
init_op = tf.global_variables_initializer()
logits = tf.placeholder(tf.float32)

def run(logits):

    with tf.Session() as sess:
        output = sess.run(tf.nn.softmax(logits))

    return output

print(output)





