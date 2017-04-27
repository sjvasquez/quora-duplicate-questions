import tensorflow as tf

from utils import shape


def cosine(a_enc, b_enc):
    """
    Compare the encoded representations a_enc and b_enc via cosine similarity.

    Args:
        a_enc: Encoded representation of sequence a. Tensor of shape [batch_size, input_units].
        b_enc: Encoded representation of sequence b. Tensor of shape [batch_size, input_units].

    Returns:
        Tensor of shape [batch size].

    """
    a_norm = tf.nn.l2_normalize(a_enc, dim=1)
    b_norm = tf.nn.l2_normalize(b_enc, dim=1)
    return tf.reduce_sum(a_norm*b_norm)


def euclidian(a_enc, b_enc):
    """
    Compare the encoded representations a_enc and b_enc via euclidian distance.

    Args:
        a_enc: Encoded representation of sequence a. Tensor of shape [batch_size, input_units].
        b_enc: Encoded representation of sequence b. Tensor of shape [batch_size, input_units].

    Returns:
        Tensor of shape [batch size].

    """
    return tf.sqrt(tf.reduce_sum(tf.square(a_enc - b_enc)), axis=1)


def manhattan(a_enc, b_enc):
    """
    Compare the encoded representations a_enc and b_enc via manhattan distance

    Args:
        a_enc: Encoded representation of sequence a. Tensor of shape [batch_size, input_units].
        b_enc: Encoded representation of sequence b. Tensor of shape [batch_size, input_units].

    Returns:
        Tensor of shape [batch size].

    """
    return tf.reduce_sum(tf.abs(a_enc - b_enc), axis=1)


def dot(a_enc, b_enc):
    """
    Compare the encoded representations a_enc and b_enc via dot product

    Args:
        a_enc: Encoded representation of sequence a. Tensor of shape [batch_size, input_units].
        b_enc: Encoded representation of sequence b. Tensor of shape [batch_size, input_units].

    Returns:
        Tensor of shape [batch size].

    """
    return tf.reduce_sum(a_enc*b_enc)


def dense(a_enc, b_enc, bias=True, activation=None, dropout=None, scope='dense', reuse=False):
    """
    Compare the encoded representations a_enc and b_enc using a learnable paramterized
    function in the form of dense layer applied to the concatenation of a_enc and b_enc.

    Args:
        a_enc: Encoded representation of sequence a. Tensor of shape [batch_size, input_units].
        b_enc: Encoded representation of sequence b. Tensor of shape [batch_size, input_units].
        activation: Activation function.
        dropout: Dropout keep prob.  Float.

    Returns:
        Tensor of shape [batch size].

    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.concat([a_enc, b_enc], axis=1)
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(inputs, -1), 1]
        )
        z = tf.matmul(inputs, W)
        if bias:
            b_enc = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[1]
            )
            z = z + b_enc
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout else z
        return tf.squeeze(z)
