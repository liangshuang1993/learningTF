import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

num_epoch = 100
length = mnist.train.images.shape[0]
print length
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])


def build_layer(input, input_size, num_neuron):
    """ build a ReLu layer"""
    weights = weight_variable([input_size, num_neuron])
    biases = bias_variable([num_neuron])
    a = tf.matmul(input, weights) + biases
    hidden = tf.nn.relu(a)
    return hidden


input_size = 784
num_neuron = [512, 128, 30]
hidden1 = build_layer(X, input_size, num_neuron[0])
input_size = num_neuron[0]
hidden2 = build_layer(hidden1, input_size, num_neuron[1])
input_size = num_neuron[1]
hidden3 = build_layer(hidden2, input_size, num_neuron[2])


weights = weight_variable([num_neuron[2], 10])
biases = bias_variable([10])

output = tf.matmul(hidden3, weights) + biases
logits = tf.nn.softmax(output)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.01, global_step, 10000, 0.9, staircase=True)
train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss,global_step=global_step)
init = tf.global_variables_initializer()    

#save the model
saver = tf.train.Saver()

plt.figure()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(num_epoch):
        for _ in range(length/100):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, l = sess.run([train_step, loss], feed_dict={X: batch_xs, y: batch_ys})
        if epoch % 10 == 0:
            print "epoch: ", epoch, "loss: ", l
        
        plt.plot(epoch, l, 'ro')
    print sess.run(accuracy, feed_dict={X: mnist.train.images, y: mnist.train.labels})
    print sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})
    plt.show()
    save_path = saver.save(sess, "./temp")


