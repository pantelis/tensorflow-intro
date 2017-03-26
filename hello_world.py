import tensorflow as tf


# basic TF operations
x = tf.constant(10, tf.int32)
y = tf.constant(2, tf.int32)
z = tf.subtract(tf.divide(x,y), 1)


with tf.Session() as sess:
    output = sess.run(z)
    print(output)
