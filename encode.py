import tensorflow as tf


def lstm_encoder(inputs, lengths, state_size, keep_prob, scope='lstm-encoder', reuse=False):
    """
    LSTM encoder

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        lengths: Tensor of shape [batch size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.

    Returns:
        Tensor of shape [batch size, state size] containing the final h states.

    """
    with tf.variable_scope(scope, reuse=reuse):
        cell_fw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
            tf.contrib.rnn.core_rnn_cell.LSTMCell(
                state_size,
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        outputs, output_state = tf.nn.dynamic_rnn(
            inputs=inputs,
            cell=cell_fw,
            sequence_length=lengths,
            dtype=tf.float32
        )
        return output_state.h


def bidirectional_lstm_encoder(inputs, lengths, state_size, keep_prob, scope='bi-lstm-encoder', reuse=False):
    """
    Bidirectional LSTM encoder

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        lengths: Tensor of shape [batch size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.

    Returns:
        Tensor of shape [batch size, 2*state size] containing theconcatenated
        forward and backward lstm final h states.

    """
    with tf.variable_scope(scope, reuse=reuse):
        cell_fw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
            tf.contrib.rnn.core_rnn_cell.LSTMCell(
                state_size,
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        cell_bw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
            tf.contrib.rnn.core_rnn_cell.LSTMCell(
                state_size,
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        outputs, (output_fw, output_bw) = tf.nn.bidirectional_dynamic_rnn(
            inputs=inputs,
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            sequence_length=lengths,
            dtype=tf.float32
        )
        outputs = tf.concat(outputs, 2)
        output_state = tf.concat([output_fw.h, output_bw.h], axis=1)
        return output_state


def reduce_max_encoder(inputs):
    """
    Max pooling over the time dimension

    Args:
        inputs: Tensor of shape [batch size, max sequence length, layer_size].

    Returns:
        Tensor of shape [batch size, layer_size].
    """
    return tf.reduce_max(inputs, axis=1)


def reduce_sum_encoder(inputs):
    """
    Sum pooling over the time dimension

    Args:
        inputs: Tensor of shape [batch size, max sequence length, layer_size].

    Returns:
        Tensor of shape [batch size, layer_size].
    """
    return tf.reduce_sum(inputs, axis=1)


def reduce_mean_encoder(inputs, lengths):
    """
    Max pooling over the time dimension

    Args:
        inputs: Tensor of shape [batch size, max sequence length, layer_size].
        lengths: Tensor of shape [batch_size] with entry i containing the integer length of sequence i

    Returns:
        Tensor of shape [batch size, layer_size].
    """
    return tf.reduce_sum(inputs, axis=1) / tf.expand_dims(lengths, 1)
