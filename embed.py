import tensorflow as tf

from encode import lstm_encoder
from utils import shape


def lstm_word_embedding(chars, lengths, embed_dim, scope='char_embed', reuse=False):
    """
    Word embeddings via LSTM encoding of character sequences.

    Args:
        chars: Tensor of shape [batch size, word sequence length, char sequence length, num characters]
        lengths: Tensor of shape [batch size, word_sequence length]
        embed_dim: dimension of word embeddings

    Returns:
        Tensor of shape [batch size, word sequence length, embed_dim]

    """
    # this is super inefficient
    chars = tf.unstack(chars, axis=1)
    lengths = tf.unstack(lengths, axis=1)

    word_embeddings = []
    for i, (char, length) in enumerate(zip(chars, lengths)):
        temp_reuse = i != 0 or reuse
        _, embedding = lstm_encoder(char, length, embed_dim, 1.0, scope=scope, reuse=temp_reuse)
        word_embeddings.append(embedding)
    word_embeddings = tf.stack(word_embeddings, axis=1)

    return word_embeddings


def convolutional_word_embedding(inputs, lengths, embed_dim, convolution_width, bias=True, scope='char_embed', reuse=False):
    """
    Word embeddings via LSTM encoding of character sequences.

    Args:
        chars: Tensor of shape [batch size, word sequence length, char sequence length, num characters]
        lengths: Tensor of shape [batch size, word_sequence length]
        embed_dim: dimension of word embeddings

    Returns:
        Tensor of shape [batch size, word sequence length, embed_dim]

    """
    with tf.variable_scope(scope, reuse=reuse):
        input_channels = shape(inputs, 1)
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[convolution_width, input_channels, embed_dim]
        )
        z = tf.nn.convolution(inputs, W, padding='SAME', strides=[1])
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[embed_dim]
            )
            z = z + b
        return z
