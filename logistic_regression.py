import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import matplotlib.pyplot as plt

length = mnist.train.images.shape[0]
n_pixels = mnist.train.images.shape[1]
n_labels = mnist.train.labels.shape[1]
learning_rate = 0.002
epoch_num = 100

W = tf.Variable(tf.zeros([n_pixels, n_labels])) 
b = tf.Variable(tf.zeros([n_labels]))
X = tf.placeholder(dtype=tf.float32, shape=[None, n_pixels], name="X")
y = tf.placeholder(dtype=tf.float32, shape=[None, n_labels], name="y")

predict = tf.matmul(X, W) + b
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

plt.figure()
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epoch_num):
        for _ in range(length/100):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, l = sess.run([train_step, loss], feed_dict={X: batch_xs, y: batch_ys})
    #        _ = sess.run([train_step], feed_dict={X: batch_xs, y: batch_ys})
        if epoch % 5 == 0:
           print "epoch=", epoch, "loss=", l
        plt.plot(epoch, l, 'ro')

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(predict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})
    plt.show()