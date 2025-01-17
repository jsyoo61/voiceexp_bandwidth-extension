import tensorflow as tf

def l1_loss(y, y_hat):

    return tf.reduce_mean(tf.abs(y - y_hat))

def l2_loss(y, y_hat):

    return tf.reduce_mean(tf.square(y - y_hat))

def sparse_categorical_cross_entropy(logits, labels):

    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels))
