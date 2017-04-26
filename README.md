# quora-duplicate-questions
detect duplicate questions on Quora

**embed.py**
  * embeddings from sparse encodings
  * word embeddings via lstm encoding of character sequences
  * word embeddings via dense layer + max pooling over character sequences
  * word embeddings via convolution + max pooling over character sequences

**propogate.py**
  * lstm layer
  * bidirectional lstm layer
  * time distributed dense layer
  * time distributed convolution layer
  * dense layer

**attend.py**
  * multiplicative attention
  * additive attention
  * concat attention
  * dot attention
  * cosine attention
  * softmax attentive matching
  * maxpool attentive matching
  * argmax attentive matching

**compare.py**
  * cosine
  * euclidian
  * manhattan
  * dot
  * dense (learnable distance function)
  * mahalanobis (with learnable covariance matrix) (TODO)
