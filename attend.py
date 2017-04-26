import tensorflow as tf

from propogate import time_distributed_dense_layer
from utils import shape


def multiplicative_attention(a, b, a_lengths, b_lengths, max_seq_len, hidden_units=150,
                             scope='multiplicative-attention', reuse=False):
    """
    For sequences a and b of lengths a_lengths and b_lengths, computes an attention matrix attn,
    where attn(i, j) = dot(W*a_i, W*b_j).  W is a learnable matrix.  The rows of attn are
    softmax normalized.

    Args:
        a: Tensor of shape [batch_size, max_seq_len, input_dim]
        b: Tensor of shape [batch_size, max_seq_len, input_dim]
        a_lengths: lengths of sequences in a of shape [batch_size]
        b_lengths: lengths of sequences in b of shape [batch_size]
        max_seq_len: length of padded sequences a and b
        hidden_units: number of hidden units

    Returns:
        Tensor of shape [max_seq_len, max_seq_len]

    """
    with tf.variable_scope(scope, reuse=reuse):
        aW = time_distributed_dense_layer(a, hidden_units, bias=False, scope='dense', reuse=False)
        bW = time_distributed_dense_layer(b, hidden_units, bias=False, scope='dense', reuse=True)
        logits = tf.matmul(aW, tf.transpose(bW, (0, 2, 1)))
        logits = logits - tf.expand_dims(tf.reduce_max(logits, axis=2), 2)
        attn = tf.exp(logits)
        attn = mask_attention_weights(attn, a_lengths, b_lengths, max_seq_len)
        return attn / tf.expand_dims(tf.reduce_sum(attn, axis=2) + 1e-10, 2)


def additive_attention(a, b, a_lengths, b_lengths, max_seq_len, hidden_units=150,
                       scope='additive-attention', reuse=False):
    """
    For sequences a and b of lengths a_lengths and b_lengths, computes an attention matrix attn,
    where attn(i, j) = dot(v, tanh(W*a_i + W*b_j)).  v is a learnable vector and W is a learnable
    matrix. The rows of attn are softmax normalized.

    Args:
        a: Tensor of shape [batch_size, max_seq_len, input_dim]
        b: Tensor of shape [batch_size, max_seq_len, input_dim]
        a_lengths: lengths of sequences in a of shape [batch_size]
        b_lengths: lengths of sequences in b of shape [batch_size]
        max_seq_len: length of padded sequences a and b
        hidden_units: number of hidden units

    Returns:
        Tensor of shape [max_seq_len, max_seq_len]

    """
    with tf.variable_scope(scope, reuse=reuse):
        aW = time_distributed_dense_layer(a, hidden_units, bias=False, scope='dense', reuse=False)
        bW = time_distributed_dense_layer(b, hidden_units, bias=False, scope='dense', reuse=True)
        aW = tf.expand_dims(aW, 2)
        bW = tf.expand_dims(bW, 1)
        v = tf.get_variable(
            name='dot_weights',
            initializer=tf.ones_initializer(),
            shape=[hidden_units]
        )
        logits = tf.einsum('ijkl,l->ijk', tf.nn.tanh(aW + bW), v)
        logits = logits - tf.expand_dims(tf.reduce_max(logits, axis=2), 2)
        attn = tf.exp(logits)
        attn = mask_attention_weights(attn, a_lengths, b_lengths, max_seq_len)
        return attn / tf.expand_dims(tf.reduce_sum(attn, axis=2) + 1e-10, 2)


def concat_attention(a, b, a_lengths, b_lengths, max_seq_len, hidden_units=150,
                     scope='concat-attention', reuse=False):
    """
    For sequences a and b of lengths a_lengths and b_lengths, computes an attention matrix attn,
    where attn(i, j) = dot(v, tanh(W*[a_i; b_j])).  v is a learnable vector and W is a learnable
    matrix.  The rows of attn are softmax normalized.

    Args:
        a: Tensor of shape [batch_size, max_seq_len, input_dim]
        b: Tensor of shape [batch_size, max_seq_len, input_dim]
        a_lengths: lengths of sequences in a of shape [batch_size]
        b_lengths: lengths of sequences in b of shape [batch_size]
        max_seq_len: length of padded sequences a and b
        hidden_units: number of hidden units

    Returns:
        Tensor of shape [max_seq_len, max_seq_len]

    """
    with tf.variable_scope(scope, reuse=reuse):
        a = tf.expand_dims(a, 2)
        b = tf.expand_dims(b, 1)
        c = tf.concat([a, b], axis=3)
        W = tf.get_variable(
            name='matmul_weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(c, -1), hidden_units]
        )
        cW = tf.einsum('ijkl,lm->ijkm', c, W)
        v = tf.get_variable(
            name='dot_weights',
            initializer=tf.ones_initializer(),
            shape=[hidden_units]
        )
        logits = tf.einsum('ijkl,l->ijk', tf.nn.tanh(cW), v)
        logits = logits - tf.expand_dims(tf.reduce_max(logits, axis=2), 2)
        attn = tf.exp(logits)
        attn = mask_attention_weights(attn, a_lengths, b_lengths, max_seq_len)
        return attn / tf.expand_dims(tf.reduce_sum(attn, axis=2) + 1e-10, 2)


def dot_attention(a, b, a_lengths, b_lengths, max_seq_len):
    """
    For sequences a and b of lengths a_lengths and b_lengths, computes an attention matrix attn,
    where attn(i, j) = dot(a_i, b_j). The rows of attn are softmax normalized.

    Args:
        a: Tensor of shape [batch_size, max_seq_len, input_dim]
        b: Tensor of shape [batch_size, max_seq_len, input_dim]
        a_lengths: lengths of sequences in a of shape [batch_size]
        b_lengths: lengths of sequences in b of shape [batch_size]
        max_seq_len: length of padded sequences a and b

    Returns:
        Tensor of shape [max_seq_len, max_seq_len]

    """
    logits = tf.matmul(a, tf.transpose(b, (0, 2, 1)))
    logits = logits - tf.expand_dims(tf.reduce_max(logits, axis=2), 2)
    attn = tf.exp(logits)
    attn = mask_attention_weights(attn, a_lengths, b_lengths, max_seq_len)
    return attn / tf.expand_dims(tf.reduce_sum(attn, axis=2) + 1e-10, 2)


def cosine_attention(a, b, a_lengths, b_lengths, max_seq_len):
    """
    For sequences a and b of lengths a_lengths and b_lengths, computes an attention matrix attn,
    where attn(i, j) = dot(a_i, b_j) / (l2_norm(a_i)*l2_norm(b_j)). The rows of attn are softmax
    normalized.

    Args:
        a: Tensor of shape [batch_size, max_seq_len, input_dim]
        b: Tensor of shape [batch_size, max_seq_len, input_dim]
        a_lengths: lengths of sequences in a of shape [batch_size]
        b_lengths: lengths of sequences in b of shape [batch_size]
        max_seq_len: length of padded sequences a and b

    Returns:
        Tensor of shape [max_seq_len, max_seq_len]

    """
    a_norm = tf.nn.l2_normalize(a, dim=2)
    b_norm = tf.nn.l2_normalize(b, dim=2)
    logits = tf.matmul(a_norm, tf.transpose(b_norm, (0, 2, 1)))
    logits = logits - tf.expand_dims(tf.reduce_max(logits, axis=2), 2)
    attn = tf.exp(logits)
    attn = mask_attention_weights(attn, a_lengths, b_lengths, max_seq_len)
    return attn / tf.expand_dims(tf.reduce_sum(attn, axis=2) + 1e-10, 2)


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


def softmax_attentive_matching(a, b, a_lengths, b_lengths, max_seq_len, attention_func=dot_attention,
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
        attention_func:  function, type of attention to use
        num_dense_layers: number of dense layers in feedforward network F
        hidden_units: number of hidden units in each layer of F

    Returns:
        Tensor of shape [batch_size, max_seq_len, input_dim] consisting of the matching vectors for
        each timestep in a.

    """
    if attention_func in {multiplicative_attention, additive_attention, concat_attention}:
        attn = attention_func(a, b, a_lengths, b_lengths, max_seq_len, hidden_units, scope=scope, reuse=reuse)
    elif attention_func in {dot_attention, cosine_attention}:
        attn = attention_func(a, b, a_lengths, b_lengths, max_seq_len)
    else:
        raise ValueError("invalid function given for attention_func")
    return tf.matmul(attn, b)


def maxpool_attentive_matching(a, b, a_lengths, b_lengths, max_seq_len, attention_func=dot_attention,
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
    if attention_func in {multiplicative_attention, additive_attention, concat_attention}:
        attn = attention_func(a, b, a_lengths, b_lengths, max_seq_len, hidden_units, scope=scope, reuse=reuse)
    elif attention_func in {dot_attention, cosine_attention}:
        attn = attention_func(a, b, a_lengths, b_lengths, max_seq_len)
    else:
        raise ValueError("invalid function given for attention_func")
    return tf.reduce_max(tf.einsum('ijk,ikl->ijkl', attn, b), axis=2)


def argmax_attentive_matching(a, b, a_lengths, b_lengths, max_seq_len, attention_func=dot_attention,
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
    if attention_func in {multiplicative_attention, additive_attention, concat_attention}:
        attn = attention_func(a, b, a_lengths, b_lengths, max_seq_len, hidden_units, scope=scope, reuse=reuse)
    elif attention_func in {dot_attention, cosine_attention}:
        attn = attention_func(a, b, a_lengths, b_lengths, max_seq_len)
    else:
        raise ValueError("invalid function given for attention_func")
    b_match_idx = tf.argmax(attn, axis=2)
    batch_index = tf.tile(tf.expand_dims(tf.range(shape(b, 0), dtype=tf.int64), 1), (1, max_seq_len))
    b_idx = tf.stack([batch_index, b_match_idx], axis=2)
    return tf.gather_nd(b, b_idx)
