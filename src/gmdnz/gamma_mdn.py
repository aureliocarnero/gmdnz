import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Input
import tensorflow.math as tfm
import tensorflow_probability as tfp
import keras
import numpy as np
import pandas as pd

tfd = tfp.distributions


def get_mixture_coef(output, tonumpy=False):
    """
    Mapping output layer to mixute components and shape
    parameters of the distribution (pi, alpha, beta)
    """
    ### split the last layer in 3 'blocks'
    out_pi, out_alpha, out_beta = tf.split(output, 3, 1)

    ### softmax transform out_pi
    max_pi = tf.reduce_max(out_pi, 1, keepdims = True)
    out_pi = tf.subtract(out_pi, max_pi)
    out_pi = tf.exp(out_pi)
    normalize_pi = tfm.reciprocal(tf.reduce_sum(out_pi, 1, keepdims = True))
    out_pi = tf.multiply(normalize_pi, out_pi)

    ### For out_alpha and out_beta we just add an offset
    alpha_min = 2.0
    out_alpha = tfm.add(out_alpha,alpha_min)

    beta_min = 0.1
    out_beta = tfm.add(out_beta,beta_min)

    if tonumpy:
        return out_pi.numpy(), out_alpha.numpy(), out_beta.numpy()
    else:
        return out_pi, out_alpha, out_beta


def gm_ll_loss(z_true, z_pred):  #This could be substituted by CRPS?
    # get mixture coefficients    
    out_pi, out_alpha, out_beta = get_mixture_coef(z_pred)

    # define mixture distribution
    gm = tfd.MixtureSameFamily(
        mixture_distribution = tfd.Categorical(probs = out_pi),
        components_distribution = tfd.Gamma(
            concentration = out_alpha,
            rate = out_beta))

    # Evaluate log-probability of y
    log_likelihood = gm.log_prob(tf.transpose(z_true))
    return -tf.reduce_mean(log_likelihood, axis = -1)




# define network architecture with Keras layers
def Gamma_MDN(K, n_neurons, losses, input_shape, RI = True):
  """
  Helper function to define and compile the MDN model.
  We allow the posibility of using the Reliability Index (RI) as metric
  """
  # Mixture parameters
  KMIX = K  # number of mixtures
  NOUT = KMIX * 3  # KMIX times a pi, alpha and beta

  optimizer = keras.optimizers.Adam(clipnorm = 1.0, learning_rate = 1e-4)

  # number of neurons of each layer
  n_hidden_1 = n_neurons  # 1st layer
  n_hidden_2 = int(n_hidden_1/2)  # 2nd layer
  n_hidden_3 = int(n_hidden_2/2)  # 3rd layer

  # set initializer properties (to fix random seed)
  initializer = keras.initializers.HeUniform(seed = 2718)

  # initialize network and add layers
  model = Sequential()
  model.add(Input(shape = (input_shape, )))
  # add hidden layer 1
  model.add(Dense(n_hidden_1, #input_shape=(input_shape,),
                  kernel_initializer = initializer,
                  use_bias = False, name = 'Hidden_1'))

  model.add(BatchNormalization(name = 'Batch_1'))
  model.add(Activation('relu', name = 'ReLU_1'))
  # add hidden layer 2
  model.add(Dense(n_hidden_2,
                  kernel_initializer = initializer,
                  use_bias = False, name = 'Hidden_2'))
  model.add(BatchNormalization(name = 'Batch_2'))
  model.add(Activation('relu', name = 'ReLU_2'))
  # add hidden layer 3
  model.add(Dense(n_hidden_3,
                  kernel_initializer = initializer,
                  use_bias = False, name = 'Hidden_3'))
  model.add(BatchNormalization(name = 'Batch_3'))
  model.add(Activation('relu', name = 'ReLU_3'))
  # add output layer (the one for the mixture parameters)
  model.add(Dense(NOUT,
                  kernel_initializer = initializer,
                  name = 'Output'))
  model.add(Activation('softplus', name = 'Softplus'))
  # compile the model
  if RI:
    model.compile(loss = losses, metrics = [RI_metric], optimizer = optimizer)
  else:
    model.compile(loss = losses, optimizer = optimizer)
  return model

def summary_model(K=4, n_neurons=128, losses=''):
    
    model = Gamma_MDN(K,n_neurons,losses, np.ones((100,10)).shape[1], RI=False)
    model.summary()


def PIT(z_true, z_pred):
  # get mixture coefficients  
  out_pi, out_alpha, out_beta = get_mixture_coef(z_pred)
  # define mixture distribution
  dist = tfd.Gamma(concentration = out_alpha, rate = out_beta)
  # calculate CDF at true redshift
  cdf = dist.cdf(z_true)
  # take weights pi into account
  M_cdf = tfm.multiply(out_pi, cdf)
  # Return PIT=sum(pi*CDF)
  return tfm.reduce_sum(M_cdf, 1, keepdims = True)

#class CheckForNaNs(keras.callbacks.Callback):
#    def on_epoch_end(self, epoch, logs=None):
#        for key, value in logs.items():
#            if isinstance(value, float) and tfm.is_nan(value):
#                print(f"NaN detected in {key} at epoch {epoch}")

def RI_metric(z_true, z_pred):
  tf.debugging.assert_all_finite(z_true, "z_true contains NaNs or Infs")
  tf.debugging.assert_all_finite(z_pred, "z_pred contains NaNs or Infs")
  # calculate PIT values
  F_t = PIT(z_true, z_pred)
  tf.debugging.assert_all_finite(F_t, "F_t contains NaNs or Infs")
  # get relative number of observations in each bin
  hist = tf.histogram_fixed_width(F_t, [0.0, 1.0], nbins = 20, dtype = tf.dtypes.int32, name = None)
  k_t = tfm.divide(hist, tfm.reduce_sum(hist))
  # calculate relative difference from uniformity
  RI = tfm.abs(tfm.subtract(k_t, 1/20))
  return tfm.reduce_sum(RI)


def pdf_mode(pis, alphas, betas):
    """ Helper function to find the mode of the mixture PDF"""

    # define mixture distribution
    gm = tfd.MixtureSameFamily(
        mixture_distribution = tfd.Categorical(probs = pis),
        components_distribution = tfd.Gamma(
            concentration = alphas,
            rate = betas))

    # compute modes of each component
    modes = (alphas - 1.0) / betas

    # find value of mixture PDF at components modes
    isFirst = True
    for i in range(len(pis[0])):
      if isFirst:
        gm_modes = gm.prob(modes[:,0])
        isFirst = False
      else:
        gm_modes = np.vstack([gm_modes, gm.prob(modes[:,i])])

    # find argument of mode giving largest PDF
    mode_arg = np.argmax(gm_modes, axis = 0)

    # find corresponding photo_z
    mode = np.ones(pis.shape[0])
    for i in range(pis.shape[0]):
      mode[i] = modes[i, mode_arg[i]]

    return mode

def return_df(pis, alphas, betas, Y_valid, Y_dnf = None, ide = None):
    """
    Given the output of the MDN, returns
    a DataFrame with mean, variance and stddev added
    and Coefficient of Variance (CoV)
    """
    pi_names = ['pi_' + str(i) for i in range(len(pis[0]))]
    alpha_names = ['alpha_' + str(i) for i in range(len(pis[0]))]
    beta_names = ['beta_' + str(i) for i in range(len(pis[0]))]
    means_names = ['mean_' + str(i) for i in range(len(pis[0]))]
    std_names = ['sdtdev_' + str(i) for i in range(len(pis[0]))]
    names = pi_names + alpha_names + beta_names + means_names + std_names
    temp = np.concatenate((pis, alphas, betas, alphas/betas, np.sqrt(alphas)/betas), axis = 1)
    df = pd.DataFrame(temp, columns = names)

    variances = alphas/betas**2
    means = (alphas / betas)
    modes = pdf_mode(pis, alphas, betas)

    df['Mode'] = modes
    df['Mean'] = np.average(means, weights = pis, axis = 1)
    df['variance'] =  np.average(means**2 + variances**2, weights = pis, axis = 1) - df['Mean'].values**2
    df['stddev'] = np.sqrt(df.variance)
    df['CoV'] = df['stddev']/df['Mean']
    df['redshift'] = Y_valid
    if Y_dnf is not None:
        df['OTHERZ'] = Y_dnf
    if ide is not None:
        df['ID'] = ide
        df.set_index('ID', inplace=True)
    return df


def model_fit(X_train, Y_train, K, n_hidden, ri_bool, verbose = 1, validation_data = None, toSave = None, n_epoch = 200, batch_size = 256*2, val_freq = 1, inloop = False, mycp = None):
    if toSave is not None and not inloop:
        mycp = keras.callbacks.ModelCheckpoint(filepath = toSave,
            save_weights_only = True,
            verbose = 0)
        cp_callback = [mycp]
    elif toSave is not None and inloop:
        cp_callback = [mycp]
    else:
        cp_callback = None

    model = Gamma_MDN(K, n_hidden, gm_ll_loss, X_train.shape[1], ri_bool)

    with tf.device('/device:GPU:0'):
        history = model.fit(X_train, Y_train,
                      validation_data = validation_data,# for first validation partition
                      #validation_data = (X_val, Y_val),# for first validation partition
                      epochs = n_epoch,
                      batch_size = batch_size,
                      verbose = verbose,
                      validation_freq = val_freq,
                      callbacks = cp_callback)
    
    return history, model, mycp
