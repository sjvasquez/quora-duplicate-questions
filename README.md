# semantic-textual-similarity
some helper functions for semantic textual similarity

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
  * global attention matrix - tanh(aWb^T)
  * factorized global attention matrix - F(a)F(b)^T, where F is a feedforward network
  * local attention matrix (TODO)
  * local attention matrix with monotonic alignment (TODO)

**match.py**
  * softmax attentive matching
  * maxpool attentive matching
  * argmax attentive matching
  * final state matching (TODO)

**compare.py**
  * cosine
  * euclidian
  * manhattan
  * dot
  * dense (learnable distance function)
  * mahalanobis (with learnable covariance matrix) (TODO)
