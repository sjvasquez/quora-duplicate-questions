import tensorflow as tf

from utils import shape


def cosine(a, b):
    """
    Compare the encoded representations of sequences a and b via cosine similarity

    Args:
        a: encoded representation of sequence a of shape [batch_size, input_units]
        b: encoded representation of sequence b of shape [batch_size, input_units]

    Returns:
        Tensor of shape [batch size]

    """
    a_norm = tf.nn.l2_normalize(a, dim=1)
    b_norm = tf.nn.l2_normalize(b, dim=1)
    return tf.reduce_sum(a_norm*b_norm)


def euclidian(a, b):
    """
    Compare the encoded representations of sequences a and b via euclidian distance

    Args:
        a: encoded representation of sequence a of shape [batch_size, input_units]
        b: encoded representation of sequence b of shape [batch_size, input_units]

    Returns:
        Tensor of shape [batch size]

    """
    return tf.sqrt(tf.reduce_sum(tf.square(a - b)), axis=1)


def manhattan(a, b):
    """
    Compare the encoded representations of sequences a and b via manhattan distance

    Args:
        a: encoded representation of sequence a of shape [batch_size, input_units]
        b: encoded representation of sequence b of shape [batch_size, input_units]

    Returns:
        Tensor of shape [batch size]

    """
    return tf.reduce_sum(tf.abs(a - b), axis=1)


def dot(a, b):
    """
    Compare the encoded representations of sequences a and b via dot product

    Args:
        a: encoded representation of sequence a of shape [batch_size, input_units]
        b: encoded representation of sequence b of shape [batch_size, input_units]

    Returns:
        Tensor of shape [batch size]

    """
    return tf.reduce_sum(a*b)


def dense(a, b, bias=True, activation=None, dropout=None, scope='dense', reuse=False):
    """
    Compare the encoded representations of sequences a and b using a learnable
    paramterized function in the form of dense layer.

    Args:
        a: encoded representation of sequence a of shape [batch_size, input_units]
        b: encoded representation of sequence b of shape [batch_size, input_units]
        activation: activation function
        dropout: dropout keep prob

    Returns:
        Tensor of shape [batch size]

    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.concat([a, b], axis=1)
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(inputs, -1), 1]
        )
        z = tf.matmul(inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[1]
            )
            z = z + b
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout else z
        return tf.squeeze(z)
