import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Input
import tensorflow.math as tfm
import tensorflow_probability as tfp
import keras
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping

tfd = tfp.distributions


def get_mixture_coef(output, tonumpy = False):
    
    """
    Maps the MDN output to the mixture coefficients (pi) and
    the Gamma distribution parameters (alpha, beta).

    Parameters:
    - output (Tensor): Network output of shape (batch_size, K*3)
    - tonumpy (bool): If True, returns NumPy arrays instead of Tensors.

    Returns:
    - out_pi: Mixture weights (softmax normalized)
    - out_alpha: Gamma shape parameters (concentration)
    - out_beta: Gamma rate parameters
    """

    # Split output into pi, alpha, beta (each of shape: batch_size x K)
    out_pi, out_alpha, out_beta = tf.split(output, 3, axis = 1)

    # Softmax normalization of pi (subtract max for numerical stability)
    max_pi = tf.reduce_max(out_pi, axis = 1, keepdims = True)
    out_pi = tf.exp(out_pi - max_pi)
    out_pi /= tf.reduce_sum(out_pi, axis = 1, keepdims = True)

    # Enforce minimum values for stability
    alpha_min = 2.0
    beta_min = 0.1
    out_alpha = out_alpha + alpha_min
    out_beta = out_beta + beta_min

    if tonumpy:
        return out_pi.numpy(), out_alpha.numpy(), out_beta.numpy()
    else:
        return out_pi, out_alpha, out_beta


def gm_ll_loss(z_true, z_pred):

    """
    Computes the negative log-likelihood loss for a Gamma Mixture Density Network.

    Parameters:
    - z_true (Tensor): True target values, shape (batch_size, 1)
    - z_pred (Tensor): Predicted mixture parameters, shape (batch_size, K*3)

    Returns:
    - loss (Tensor): Mean negative log-likelihood
    """

    # Extract mixture coefficients and distribution parameters
    out_pi, out_alpha, out_beta = get_mixture_coef(z_pred)

    # Define the Gamma mixture model
    gm = tfd.MixtureSameFamily(
        mixture_distribution = tfd.Categorical(probs = out_pi),
        components_distribution = tfd.Gamma(
            concentration = out_alpha,
            rate = out_beta)
        )

    # Calculate log-probability of true values
    log_likelihood = gm.log_prob(tf.squeeze(z_true, axis = -1))  # Ensure proper shape

    # Return mean negative log-likelihood
    return -tf.reduce_mean(log_likelihood)


def crps_mixture_gamma(z_true, z_pred, num_samples = 200):

    """
    Computes CRPS for a mixture of Gamma distributions using Monte Carlo approximation.

    Parameters:
    - z_true: Tensor of shape (batch_size, 1), true target values.
    - z_pred: Tensor of shape (batch_size, K*3), MDN predictions.
    - num_samples: Number of Monte Carlo samples to approximate CRPS.

    Returns:
    - crps: Tensor, mean CRPS over the batch.
    """

    batch_size = tf.shape(z_true)[0]
    out_pi, out_alpha, out_beta = get_mixture_coef(z_pred)

    # Define Gamma Mixture
    gm = tfd.MixtureSameFamily(
        mixture_distribution = tfd.Categorical(probs = out_pi),
        components_distribution = tfd.Gamma(concentration = out_alpha, rate = out_beta)
    )

    # Sample from the predicted mixture
    samples = gm.sample(num_samples)  # Shape: (num_samples, batch_size)
    samples = tf.transpose(samples)   # Shape: (batch_size, num_samples)

    # First CRPS term: Mean absolute error between samples and true value
    abs_diff = tf.abs(samples - z_true)
    term1 = tf.reduce_mean(abs_diff, axis = 1)  # Shape: (batch_size,)

    # Second CRPS term: Expected pairwise distance between samples
    samples1 = tf.expand_dims(samples, axis = 2)  # Shape: (batch_size, num_samples, 1)
    samples2 = tf.expand_dims(samples, axis = 1)  # Shape: (batch_size, 1, num_samples)
    pairwise_diff = tf.abs(samples1 - samples2)
    term2 = 0.5 * tf.reduce_mean(pairwise_diff, axis = [1, 2])  # Shape: (batch_size,)

    crps = term1 - term2
    return tf.reduce_mean(crps)  # Return mean CRPS over batch


def Gamma_MDN(K, n_neurons, losses, input_shape, RI = True):

    """
    Builds and compiles a Gamma Mixture Density Network (MDN) using Keras Sequential API.

    Parameters:
    - K (int): Number of mixture components.
    - n_neurons (int): Number of neurons in the first hidden layer.
    - losses (str or callable): Loss function to use.
    - input_shape (int): Number of input features.
    - RI (bool): Whether to include the Reliability Index (RI) as a metric.

    Returns:
    - model (keras.Model): Compiled Keras model.
    """

    # Mixture model parameters: K components, each with pi, alpha, and beta
    KMIX = K  
    NOUT = KMIX * 3  # Total outputs: pi, alpha, beta for each mixture component

    optimizer = keras.optimizers.Adam(clipnorm = 1.0, learning_rate = 1e-4)

    # Layer sizes
    n_hidden_1 = n_neurons
    n_hidden_2 = n_hidden_1 // 2
    n_hidden_3 = n_hidden_2 // 2

    # Weight initializer
    initializer = keras.initializers.HeUniform(seed = 2718)

    # Build the model
    model = Sequential()
    model.add(Input(shape=(input_shape, ), name = 'Input'))

    # Hidden Layer 1
    model.add(Dense(n_hidden_1, kernel_initializer = initializer, use_bias = False, name = 'Hidden_1'))
    model.add(BatchNormalization(name = 'Batch_1'))
    model.add(Activation('relu', name = 'ReLU_1'))

    # Hidden Layer 2
    model.add(Dense(n_hidden_2, kernel_initializer = initializer, use_bias = False, name = 'Hidden_2'))
    model.add(BatchNormalization(name = 'Batch_2'))
    model.add(Activation('relu', name = 'ReLU_2'))

    # Hidden Layer 3
    model.add(Dense(n_hidden_3, kernel_initializer = initializer, use_bias = False, name = 'Hidden_3'))
    model.add(BatchNormalization(name = 'Batch_3'))
    model.add(Activation('relu', name = 'ReLU_3'))

    # Output Layer: Mixture Parameters
    model.add(Dense(NOUT, kernel_initializer = initializer, name = 'Output'))
    model.add(Activation('softplus', name = 'Softplus'))

    # Compile the model
    if RI:
        model.compile(loss = losses, metrics = [RI_metric], optimizer = optimizer)
    else:
        model.compile(loss = losses, optimizer = optimizer)

    return model


def summary_model(K = 4, n_neurons = 128, losses = ''):
    
    """
    Prints the summary of the Gamma-MDN model.
    """

    model = Gamma_MDN(K, n_neurons, losses, 10, RI = False)
    model.summary()


def PIT(z_true, z_pred):

    """
    Computes the Probability Integral Transform (PIT) values for a gamma mixture model.
    """

    out_pi, out_alpha, out_beta = get_mixture_coef(z_pred)

    # Define the gamma mixture components
    dist = tfd.Gamma(concentration = out_alpha, rate = out_beta)

    # Calculate CDF at true redshift
    cdf = dist.cdf(z_true)

    # Weight by mixture probabilities
    weighted_cdf = tfm.multiply(out_pi, cdf)

    # Return PIT as the sum over mixture components
    return tfm.reduce_sum(weighted_cdf, axis = 1, keepdims = True)


def RI_metric(z_true, z_pred):

    """
    Computes the Relative Index (RI) metric, which quantifies the deviation of PIT histogram from uniformity.
    """

    tf.debugging.assert_all_finite(z_true, "z_true contains NaNs or Infs")
    tf.debugging.assert_all_finite(z_pred, "z_pred contains NaNs or Infs")

    # Calculate PIT values
    F_t = PIT(z_true, z_pred)
    tf.debugging.assert_all_finite(F_t, "F_t contains NaNs or Infs")

    # Compute histogram (number of observations in each bin)
    hist = tf.histogram_fixed_width(F_t, [0.0, 1.0], nbins = 20, dtype = tf.int32)

    # Convert counts to relative frequencies
    k_t = tfm.divide(hist, tfm.reduce_sum(hist))

    # Compute the absolute difference from uniformity (ideal is 1/20 per bin)
    RI = tfm.abs(k_t - (1.0 / 20))

    # Sum of deviations gives the RI metric
    return tfm.reduce_sum(RI)


def pdf_mode(pis, alphas, betas):
    
    """
    Compute the mode (most probable value) of a Gamma Mixture Model (GMM)
    for each sample.

    Parameters
    ----------
    pis : np.array, shape (n_samples, K)
        Mixture weights for each component.
    alphas : np.array, shape (n_samples, K)
        Shape parameters of the Gamma distributions.
    betas : np.array, shape (n_samples, K)
        Rate parameters of the Gamma distributions.

    Returns
    -------
    mode : np.array, shape (n_samples,)
        Mode of the mixture PDF for each sample.
    """

    # Define the mixture distribution
    gm = tfd.MixtureSameFamily(
            mixture_distribution = tfd.Categorical(probs = pis),
            components_distribution = tfd.Gamma(
                concentration = alphas,
                rate = betas)
        )

    # Compute modes of individual Gamma components
    modes = (alphas - 1.0) / betas

    # Evaluate the mixture PDF at each component's mode
    gm_modes = np.vstack([gm.prob(modes[:, i]) for i in range(pis.shape[1])])

    # Find which component's mode has the highest PDF value for each sample
    mode_arg = np.argmax(gm_modes, axis = 0)

    # Retrieve the mode corresponding to the maximum PDF for each sample
    mode = modes[np.arange(pis.shape[0]), mode_arg]

    return mode


def return_df(pis, alphas, betas, Y_valid, Y_dnf = None, ide = None):

    """
    Build a DataFrame summarizing the mixture components, means, variances,
    standard deviations, and additional statistics from an MDN output.

    Parameters
    ----------
    pis : np.array
        Mixture weights, shape (n_samples, K).
    alphas : np.array
        Gamma distribution alpha parameters, shape (n_samples, K).
    betas : np.array
        Gamma distribution beta parameters, shape (n_samples, K).
    Y_valid : np.array
        True redshift values for the validation/test set.
    Y_dnf : np.array, optional
        Optional secondary redshift estimates (e.g., from another method).
    ide : np.array or pd.Series, optional
        Optional array of unique IDs to use as the DataFrame index.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing component parameters, means, variances, modes,
        standard deviations, coefficient of variation (CoV), and true redshift values.
    """

    K = pis.shape[1]

    # Build dynamic column names
    pi_names = [f'pi_{i}' for i in range(K)]
    alpha_names = [f'alpha_{i}' for i in range(K)]
    beta_names = [f'beta_{i}' for i in range(K)]
    means_names = [f'mean_{i}' for i in range(K)]
    std_names = [f'stddev_{i}' for i in range(K)]

    all_column_names = pi_names + alpha_names + beta_names + means_names + std_names

    # Calculate means and standard deviations for each component
    component_means = alphas / betas
    component_stds = np.sqrt(alphas) / betas

    # Build the base DataFrame
    temp = np.concatenate((pis, alphas, betas, component_means, component_stds), axis = 1)
    df = pd.DataFrame(temp, columns = all_column_names)

    # Global statistics per sample
    variances = alphas / betas**2
    modes = pdf_mode(pis, alphas, betas)

    df['Mode'] = modes
    df['Mean'] = np.average(component_means, weights = pis, axis = 1)
    df['variance'] = np.average(component_means**2 + variances, weights = pis, axis = 1) - df['Mean']**2
    df['stddev'] = np.sqrt(df['variance'])
    df['CoV'] = df['stddev'] / df['Mean']

    # Add true redshift
    df['redshift'] = Y_valid

    # Optional: Add secondary estimates or ID index
    if Y_dnf is not None:
        df['OTHERZ'] = Y_dnf

    if ide is not None:
        df['ID'] = ide
        df.set_index('ID', inplace = True)

    return df


def model_fit(X_train, Y_train, K, n_hidden, ri_bool, verbose = 1, validation_data = None, 
        toSave = None, n_epoch = 200, batch_size = 512, val_freq = 1, lossF = 'll', 
        inloop = False, mycp = None, early_stop = False):

    """
    Train a Gamma-MDN model with optional checkpointing.

    Parameters
    ----------
    X_train : np.array
        Training features.
    Y_train : np.array
        Training labels.
    K : int
        Number of mixture components.
    n_hidden : list
        List specifying the number of hidden units per layer.
    ri_bool : bool
        Whether to include Random Initialization regularization.
    verbose : int, optional
        Verbosity level for training output.
    validation_data : tuple, optional
        Validation dataset (X_val, Y_val).
    toSave : str, optional
        File path to save model checkpoints.
    n_epoch : int, optional
        Number of training epochs.
    batch_size : int, optional
        Training batch size.
    val_freq : int, optional
        Frequency (in epochs) of validation evaluation.
    lossF : str, optional
        Loss function, ll = log-likelihood (gm_ll_loss), crps = CRPS (crps_mixture_gamma)
    inloop : bool, optional
        If True, continues using a checkpoint provided in `mycp`.
    mycp : keras.callbacks.ModelCheckpoint, optional
        Existing checkpoint callback to use if inloop is True.
    early_stop : bool, optional
        If True, early stp based on val_loss. Setting val to 0.1 if not present

    Returns
    -------
    history : keras.callbacks.History
        Training history object.
    model : keras.Model
        Trained Gamma-MDN model.
    mycp : keras.callbacks.ModelCheckpoint
        Model checkpoint callback.
    """

    # Setup model checkpointing
    if toSave is not None and not inloop:
        mycp = keras.callbacks.ModelCheckpoint(filepath = toSave,
                                        save_weights_only = True,
                                        verbose = 0)
        cp_callback = [mycp]
    elif toSave is not None and inloop:
        cp_callback = [mycp]
    else:
        cp_callback = None


    if lossF == 'll':
        loss_function = gm_ll_loss
    elif lossF == 'crps':
        loss_function = crps_mixture_gamma
    else:
        raise Exception('Loss function not correctly set')

    if early_stop:
        # Define the early stopping callback
        early_stop = EarlyStopping(
            min_delta = 0.001,
            monitor = 'val_loss',       # What to monitor
            patience = 5,               # Number of epochs with no improvement after which training will be stopped
            restore_best_weights = True # Restore model weights from the epoch with the best monitored value
        )
        if cp_callback is None:
            cp_callback = [early_stop]
        else:
            cp_callback.append(early_stop)

        validation_split = 0.1
    else:
        validation_split = 0.

    # Instantiate the model
    model = Gamma_MDN(K, n_hidden, loss_function, X_train.shape[1], ri_bool)
    

    # Train the model using GPU
    with tf.device('/device:GPU:0'):
        history = model.fit(X_train, Y_train,
                      validation_data = validation_data,# for first validation partition
                      validation_split = validation_split,
                      epochs = n_epoch,
                      batch_size = batch_size,
                      verbose = verbose,
                      validation_freq = val_freq,
                      callbacks = cp_callback)
    
    return history, model, mycp
