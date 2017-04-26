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
  * multiplicative attention: attn(i, j) = dot(dot(v, tanh(Wa<sub>i</sub>)), dot(v, tanh(Wb<sub>j</sub>)))
  * 
  * additive attention: attn(i, j) = dot(v, tanh(Wa<sub>i</sub> + Wb<sub>j</sub>))
  * concat attention: attn(i, j) = dot(v, tanh(W[a<sub>i</sub>; b<sub>j</sub>]))
  * dot attention: attn(i, j) = dot(a<sub>i</sub>, b<sub>j</sub>)
  * cosine attention: attn(i, j) = cosine(a<sub>i</sub>, b<sub>j</sub>)

**match.py**
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
