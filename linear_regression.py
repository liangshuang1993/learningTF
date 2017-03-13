import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

learning_rate = 0.01
training_epochs = 1000
display_step = 50

train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

X = tf.placeholder("float32")
Y = tf.placeholder("float32")
W = tf.Variable(tf.constant(0.), name="weight")
b = tf.Variable(tf.constant(0.), name="bias")
tf.summary.scalar('W', W)
tf.summary.histogram('histogram', W)

pred = X * W + b
cost = tf.reduce_sum(tf.pow(pred - Y, 2))/(2 * n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

plt.figure(1)
"""
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print "Epoch", '%04d'%(epoch + 1), "cost=", "{:.9f}".format(c),\
            "W=", sess.run(W), "b=", sess.run(b)

    print "Finished"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost
    plt.plot(train_X, train_Y, 'ro', label = "original data")
    plt.plot(train_X, sess.run(pred, feed_dict={X: train_X}), label = "fitted line")
    plt.legend()
    plt.show()
"""
with tf.Session() as sess:
    train_writter = tf.summary.FileWriter('./board', sess.graph)
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if (epoch + 1) % display_step == 0:
            summary, acc = sess.run([merged, cost], feed_dict={X: train_X, Y: train_Y})

