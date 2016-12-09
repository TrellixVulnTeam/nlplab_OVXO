import tensorflow as tf

with tf.Session() as sess:
    sess_2 = tf.get_default_session()

sess_2.close()