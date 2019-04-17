from keras.layers import Dense, Dropout
import keras.backend as K

class EstimationNet:
  """ Estimation network for the GMM model"""
  def __init__(self, layer_sizes, num_comp, activation=K.tanh):
    self.layer_sizes = layer_sizes
    self.activation  = activation
    self.num_comp     = num_comp

  def inference(self, model_input, dropout_ratio=0.0):
    """ Output softmax probabilities from the estimation net. """
    z = model_input

    n_layer = 0
    for size in self.layer_sizes:
      n_layer += 1
      z = Dense(size, activation=self.activation, 
                name=f'estnet_layer_{n_layer}')(z)
      if (dropout_ratio > 0.0):
        z = Dropout(dropout_ratio)(z)

    size = self.num_comp
    output = Dense(size, activation='softmax', name='gamma')(z)

    return output
