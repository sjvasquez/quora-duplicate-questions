import tensorflow as tf

from attend import attention_matrix, factorized_attention_matrix
from utils import shape


def softmax_attentive_matching(a, b, a_lengths, b_lengths, max_seq_len, factorized=True, num_dense_layers=1,
                               hidden_units=150, scope='softmax_attentive-matching', reuse=False):
    """
    Matches each vector in a with a weighted sum of the vectors in b.  The weighted sum is determined
    by the attention matrix.  If factorized is True, then the attention matrix is the calculated
    using factorized_attention_matrix.  Otherwise, attention_matrix will be used.

    Args:
        a: Tensor of shape [batch_size, max_seq_len, input_dim]
        b: Tensor of shape [batch_size, max_seq_len, input_dim]
        a_lengths: lengths of sequences in a of shape [batch_size]
        b_lengths: lengths of sequences in b of shape [batch_size]
        max_seq_len: length of padded sequences a and b
        factorized:  Whether or not to use a factorized attention matrix
        num_dense_layers: number of dense layers in feedforward network F
        hidden_units: number of hidden units in each layer of F

    Returns:
        Tensor of shape [batch_size, max_seq_len, input_dim] consisting of the matching vectors for
        each timestep in a.

    """
    if factorized:
        attn = factorized_attention_matrix(a, b, a_lengths, b_lengths, max_seq_len, num_dense_layers,
                                           hidden_units, scope=scope, reuse=reuse)
    else:
        attn = attention_matrix(a, b, a_lengths, b_lengths, max_seq_len, scope=scope, reuse=reuse)
    return tf.matmul(attn, b)


def maxpool_attentive_matching(a, b, a_lengths, b_lengths, max_seq_len, factorized=True, num_dense_layers=1,
                               hidden_units=150, scope='maxpool_attentive-matching', reuse=False):
    """
    Same as softmax_attentive_matching, but after weighting elements in b, uses maxpooling to reduce
    the dimension instead of contraction.

    Args:
        a: Tensor of shape [batch_size, max_seq_len, input_dim]
        b: Tensor of shape [batch_size, max_seq_len, input_dim]
        a_lengths: lengths of sequences in a of shape [batch_size]
        b_lengths: lengths of sequences in b of shape [batch_size]
        max_seq_len: length of padded sequences a and b
        factorized:  Whether or not to use a factorized attention matrix
        num_dense_layers: number of dense layers in feedforward network F
        hidden_units: number of hidden units in each layer of F

    Returns:
        Tensor of shape [batch_size, max_seq_len, input_dim] consisting of the matching vectors for
        each timestep in a.

    """
    if factorized:
        attn = factorized_attention_matrix(a, b, a_lengths, b_lengths, max_seq_len, num_dense_layers,
                                           hidden_units, scope=scope, reuse=reuse)
    else:
        attn = attention_matrix(a, b, a_lengths, b_lengths, max_seq_len, scope=scope, reuse=reuse)
    return tf.reduce_max(tf.einsum('ijk,ikl->ijkl', attn, b), axis=2)


def argmax_attentive_matching(a, b, a_lengths, b_lengths, max_seq_len, factorized=True, num_dense_layers=1,
                              hidden_units=150, scope='argmax_attentive-matching', reuse=False):
    """
    Same as softmax_attentive_matching, but instead of matching each vector in a with a weighted sum
    of the vectors in b, it simply uses the most similar vector in b, where similarity is defined using
    the values in the attention matrix.

     Args:
        a: Tensor of shape [batch_size, max_seq_len, input_dim]
        b: Tensor of shape [batch_size, max_seq_len, input_dim]
        a_lengths: lengths of sequences in a of shape [batch_size]
        b_lengths: lengths of sequences in b of shape [batch_size]
        max_seq_len: length of padded sequences a and b
        factorized:  Whether or not to use a factorized attention matrix
        num_dense_layers: number of dense layers in feedforward network F
        hidden_units: number of hidden units in each layer of F

    Returns:
        Tensor of shape [batch_size, max_seq_len, input_dim] consisting of the matching vectors for
        each timestep in a.

    """
    attn_logits = tf.matmul(a, tf.transpose(b, (0, 2, 1)))
    b_match_idx = tf.argmax(attn_logits, axis=2)
    batch_index = tf.tile(tf.expand_dims(tf.range(shape(b, 0), dtype=tf.int64), 1), (1, max_seq_len))
    b_idx = tf.stack([batch_index, b_match_idx], axis=2)
    return tf.gather_nd(b, b_idx)
