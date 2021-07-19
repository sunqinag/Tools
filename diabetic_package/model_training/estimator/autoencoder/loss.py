import tensorflow as tf

def euclidean_distance_loss(inputs, logits):
    regularization = tf.losses.get_regularization_loss()
    loss = tf.reduce_mean(tf.pow(logits - inputs, 2)) + regularization

    return loss

def custom_loss(inputs, logits):
    inputs_sign = tf.sign(inputs)
    Intersection = tf.multiply(logits, inputs_sign)
    loss = (tf.reduce_sum(tf.square(logits-inputs)) + 1e-5)/\
           (tf.reduce_sum(tf.square(logits)) +\
                            tf.reduce_sum(tf.square(inputs)) +\
                            tf.reduce_sum(tf.square(Intersection)) + 1e-5)
    return loss


def l2_loss(inputs, logits):
    # return 2 * tf.nn.l2_loss(imgs_true - imgs_pred)
    return tf.nn.l2_loss(inputs - logits)