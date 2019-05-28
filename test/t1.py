import tensorflow as tf

m1 = tf.constant([3, 5])
m2 = tf.constant([2, 4])

result = tf.add(m1, m2)

print(result)

with tf.Session() as sess:
    print(sess.run(result))
