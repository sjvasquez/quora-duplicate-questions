import tensorflow as tf

from propogate import time_distributed_dense
from utils import shape


def mask_attention_weights(weights, a_lengths, b_lengths, max_seq_len):
    """
    Masks an attention matrix for sequences a and b of lengths a_lengths and b_lengths so that
    the attention matrix of shape max_len by max_len contains zeros outside of
    a_lengths by b_lengths submatrix int the top left corner.

    Args:
        weights: Tensor of shape [max_seq_len, max_seq_len].
        a_lengths: Lengths of sequences in a of shape [batch size].
        b_lengths: Lengths of sequences in b of shape [batch size].
        max_seq_len: Max length of padded sequences

    Returns:
        Tensor of shape [max_seq_len, max_seq_len]

    """
    a_mask = tf.expand_dims(tf.sequence_mask(a_lengths, maxlen=max_seq_len), 2)
    b_mask = tf.expand_dims(tf.sequence_mask(b_lengths, maxlen=max_seq_len), 1)
    seq_mask = tf.cast(tf.matmul(tf.cast(a_mask, tf.int32), tf.cast(b_mask, tf.int32)), tf.bool)
    return tf.where(seq_mask, weights, tf.zeros_like(weights))


def attention_matrix(a, b, a_lengths, b_lengths, max_seq_len, scope='attention', reuse=False):
    """
    For sequences a and b of lengths a_lengths and b_lengths, computes an attention matrix of the
    form tanh(a*W*b^T) where W is a trainable parameter matrix.

    Args:
        a: Tensor of shape [batch_size, max_seq_len, input_dim]
        b: Tensor of shape [batch_size, max_seq_len, input_dim]
        a_lengths: lengths of sequences in a of shape [batch_size]
        b_lengths: lengths of sequences in b of shape [batch_size]
        max_seq_len: length of padded sequences a and b

    Returns:
        Tensor of shape [max_seq_len, max_seq_len]

    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(a, -1), shape(b, -1)]
        )
        aWb = tf.nn.tanh(tf.matmul(tf.matmul(a, W), tf.transpose(b, (0, 2, 1))))
        logits = aWb - tf.expand_dims(tf.reduce_max(aWb, axis=2), 2)
        weights = tf.exp(logits)
        weights = mask_attention_weights(weights, a_lengths, b_lengths, max_seq_len)
        return weights / tf.expand_dims(tf.reduce_sum(weights, axis=2) + 1e-10, 2)


def factorized_attention_matrix(a, b, a_lengths, b_lengths, max_seq_len, num_dense_layers=1, hidden_units=150,
                                scope='factorized-attention', reuse=False):
    """
    For sequences a and b of lengths a_lengths and b_lengths, computes an attention matrix of the
    form F(a)*F(b)^T, where F is a feedforward network.

    Args:
        a: Tensor of shape [batch_size, max_seq_len, input_dim]
        b: Tensor of shape [batch_size, max_seq_len, input_dim]
        a_lengths: lengths of sequences in a of shape [batch_size]
        b_lengths: lengths of sequences in b of shape [batch_size]
        max_seq_len: length of padded sequences a and b
        num_dense_layers: number of dense layers in feedforward network F
        hidden_units: number of hidden units in each layer of F

    Returns:
        Tensor of shape [max_seq_len, max_seq_len]

    """
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(num_dense_layers):
            activation = None if i == num_dense_layers - 1 else tf.nn.tanh
            a = time_distributed_dense(a, hidden_units, activation=activation,
                                       bias=False, scope='dense_' + str(i), reuse=False)
            b = time_distributed_dense(b, hidden_units, activation=activation,
                                       bias=False, scope='dense_' + str(i), reuse=True)
        logits = tf.matmul(a, tf.transpose(b, (0, 2, 1)))
        logits = logits - tf.expand_dims(tf.reduce_max(logits, axis=2), 2)
        weights = tf.exp(logits)
        weights = mask_attention_weights(weights, a_lengths, b_lengths, max_seq_len)
        return weights / tf.expand_dims(tf.reduce_sum(weights, axis=2) + 1e-10, 2)
