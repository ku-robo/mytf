import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = 0.1 * x_data + 0.3

# Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
Weights = tf.Variable(tf.zeros([1]))
Weights2 = tf.Variable(tf.zeros([1]))
biases = tf.Variable(tf.zeros([1]))

y = Weights2 * x_data * x_data + Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(Weights2), sess.run(Weights), sess.run(biases), sess.run(loss))
