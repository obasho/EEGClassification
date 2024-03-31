import tensorflow as tf
def triplet_loss(y_true, y_pred, margin=0.5):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Compute Euclidean distances between the anchor and the positive/negative examples
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

    # Compute loss
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)

    # Sum over all triplets
    loss = tf.reduce_sum(loss)

    return loss