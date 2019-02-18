from keras.layers import Dense, Dropout
import keras.backend as K

class EstimationNet:
  """ Estimation network for the GMM model"""
  def __init__(self, layer_sizes, activation=K.tanh):
    self.layer_sizes = layer_sizes
    self.activation  = activation

  def inference(self, model_input, dropout_ratio=0.0):
    """ Output softmax probabilities from the estimation net. """
    z = model_input

    n_layer = 0
    for size in self.layer_sizes[:-1]:
      n_layer += 1
      print ("est_size", size, type(size))
      z = Dense(size, activation=self.activation, 
                name=f'estnet_layer_{n_layer}')(z)
      if (dropout_ratio > 0.0):
        z = Dropout(dropout_ratio)(z)

    size = self.layer_sizes[-1]
    output = Dense(size, activation='softmax', name='gamma')(z)

    return output
