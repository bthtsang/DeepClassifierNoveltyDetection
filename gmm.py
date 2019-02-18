import numpy as np
import tensorflow as tf
import keras.backend as K

def add_noise(mat):
  """
  :param mat: should be of shape(k, d, d)
  :return: a matrix with little noises
  """
  sml_value = 1.0e-4
  n_features = mat.get_shape().as_list()[1]
  noise = tf.matrix_diag(tf.ones(n_features, dtype=tf.float64)) * sml_value
  noise = tf.expand_dims(noise, axis=0)
  noise = tf.tile(noise, (mat.get_shape()[0], 1, 1))

  return mat + noise


class GMM: 
  """ Gaussian Mixture Model (GMM) """
  def __init__(self, n_comp, n_features):
    self.n_comp = n_comp
    self.n_features = n_features
    # Setting GMM parameters to stub
    self.phi = self.mu = self.sigma = None
    self.fitted = False

  ### Create variables to store the GMM model values
  def create_variables(self, n_features):
    phi = K.variable(np.zeros(shape=[self.n_comp]),
                     dtype="float64", name="phi")
    mu = K.variable(np.zeros(shape=[self.n_comp, n_features]),
                dtype="float64", name="mu")
    sigma = K.variable(np.zeros(
                shape=[self.n_comp, n_features, n_features]),
                dtype="float64", name="sigma")
    L = K.variable(np.zeros(
                shape=[self.n_comp, n_features, n_features]),
                dtype="float64", name="L")
    return phi, mu, sigma, L

  def init_gmm_variables(self):
    self.phi = tf.zeros(shape=[self.n_comp], 
                        dtype="float64", name='phi')
    self.mu  = tf.zeros(shape=[self.n_comp, self.n_features], 
                        dtype="float64", name='mu')
    mat_id = tf.matrix_diag(tf.ones(self.n_features, dtype=tf.float64))
    mat_id = tf.expand_dims(mat_id, axis=0)
    mat_id = tf.tile(mat_id, (self.n_comp, 1, 1))
    self.sigma = mat_id
    return 0

  def fit(self, z, gamma):
    """ GMM model parameter fitting"""
    # Compute phi
    gamma_sum = tf.reduce_sum(gamma, axis=0) # dim = (K)
    self.phi = phi = tf.reduce_mean(gamma, axis=0) # dim = (K)

    gamma_sum_exp_dims = tf.expand_dims(gamma_sum, axis=-1) # dim = (1)
    self.mu = mu = tf.matmul(gamma, z, transpose_a=True) / gamma_sum_exp_dims # dim = (K, embedding)

    phi_exp_dims = tf.expand_dims(self.phi, axis=0)
    phi_exp_dims = tf.expand_dims(phi_exp_dims, axis=-1)
    phi_exp_dims = tf.expand_dims(phi_exp_dims, axis=-1)

    z_exp_dims = tf.expand_dims(z, 1)
    z_exp_dims = tf.expand_dims(z_exp_dims, -1)
    mu_exp_dims = tf.expand_dims(self.mu, 0)
    mu_exp_dims = tf.expand_dims(mu_exp_dims, -1)

    z_minus_mu = z_exp_dims - mu_exp_dims

    sigma = tf.matmul(z_minus_mu, z_minus_mu, transpose_b=True)
    broadcast_gamma = tf.expand_dims(gamma, axis=-1)
    broadcast_gamma = tf.expand_dims(broadcast_gamma, axis=-1)
    sigma = broadcast_gamma * sigma
    sigma_i = sigma # sigma_i has dimensions (N, K, D, D)

    sigma = tf.reduce_sum(sigma, axis=0)
    sigma = sigma / tf.expand_dims(gamma_sum_exp_dims, axis=-1)
    sigma = add_noise(sigma)
    self.sigma = sigma

    self.fitted = True
    return sigma_i

  def energy(self, z, sigma_i):
    """ GMM sample energy calculation """ 

    phi_exp_dims = tf.expand_dims(self.phi, axis=0)
    phi_exp_dims = tf.expand_dims(phi_exp_dims, axis=-1)
    phi_exp_dims = tf.expand_dims(phi_exp_dims, axis=-1)

    z_exp_dims = tf.expand_dims(z, 1)
    z_exp_dims = tf.expand_dims(z_exp_dims, -1)
    mu_exp_dims = tf.expand_dims(self.mu, 0)
    mu_exp_dims = tf.expand_dims(mu_exp_dims, -1)

    z_centered = z_exp_dims - mu_exp_dims

    sigma_inverse = tf.expand_dims(tf.matrix_inverse(self.sigma), axis=0)
    sigma_inverse = tf.tile(sigma_inverse, [tf.shape(z_centered)[0], 1, 1, 1])
    
    energies = tf.matmul(z_centered, sigma_inverse, transpose_a=True)
    energies = tf.matmul(energies, z_centered)
    energies = tf.squeeze(phi_exp_dims * tf.exp(-0.5*energies), axis=[2, 3])

    energies_divided_by = tf.expand_dims(tf.sqrt(2.0*np.pi*tf.matrix_determinant(self.sigma)), axis=0) + 1e-12
    energies = tf.reduce_sum(energies / energies_divided_by, axis=1) + 1e-12
    energies = -1.0*tf.log(energies)

    energies = energies[:, None]
    self.fitted = False

    return energies


  def cov_diag_loss(self):
    diag_loss = tf.reduce_sum(tf.divide(1, tf.matrix_diag_part(self.sigma)))
    return diag_loss
