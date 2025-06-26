import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import pandas as pd
import gamma_mdn as GM

def plot_gamma_mix(pis, alphas, betas, ax = None, color = 'red', maxredshift = 2., bins = 600, namesave = 'plot_gamma_mix'):
    
    """
    Plot a mixture of Gamma distributions.

    This function visualizes the probability density function (PDF) of a Gamma mixture model,
    representing the multimodal distributions in photometric redshift space.

    Parameters
    ----------
    pis : array-like
        Mixture weights for each Gamma component. Must sum to 1.
    alphas : array-like
        Shape parameters (concentration) of the Gamma components.
    betas : array-like
        Rate parameters of the Gamma components.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis object to plot on. If None, a new figure is created.
    color : str, optional
        Color of the plotted curve. Default is 'red'.
    maxredshift : float, optional
        Upper limit of the x-axis (maximum redshift). Default is 2.0.
    bins : int, optional
        Number of points used to discretize the x-axis. Default is 600.
    namesave : str, optional
        Base name for saving the plot as a PNG file if `ax` is not provided. Default is 'plot_gamma_mix'.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis object with the plot, if `ax` is provided.

    Notes
    -----
    If no axis is provided, the plot will be automatically saved to a PNG file.
    """
    
    # Create x-axis points
    x = np.linspace(0.0, maxredshift, bins)
    final = np.zeros_like(x)

    # Sum the weighted PDFs of each Gamma component
    for i, (weight_mix, alpha_mix, beta_mix) in enumerate(zip(pis, alphas, betas)):
        dist = tfd.Gamma(concentration = alpha_mix, rate = beta_mix)
        pdf = dist.prob(x)
        temp = pdf.numpy() * weight_mix
        final = final + temp

    # Create plot if axis is not provided
    if ax == None:
        fig, ax = plt.subplots()

    ax.plot(x, final, c = color)

    # Save figure if axis was created internally
    if ax == None:
        plt.savefig(namesave + '.png')
        plt.close()
    else:
        return ax


def plot_variables(data, feat_disp, namesave = 'plot_variables'):

    """
    Plot histograms of selected features.

    This function generates side-by-side histograms (density plots) for a list of features
    from the provided dataset. It is useful for quickly visualizing the distribution
    of multiple variables.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing the features to plot.
    feat_disp : list of str
        List of column names in `data` to display.
    namesave : str, optional
        Base name for saving the plot as a PNG file. Default is 'plot_variables'.

    Notes
    -----
    The histograms are plotted with density normalization and step-style bins.
    The final figure is automatically saved to disk.
    """

    # Set up a row of subplots, one for each feature
    fig, axes = plt.subplots(nrows = 1, ncols = len(feat_disp), figsize = (20,3))

    for i in range(int(len(feat_disp))):
        # Plot histogram with density normalization
        sns.histplot(data[feat_disp[i]],
                 element = "step",
                 stat = "density",
                 bins = int(180/5),
                 color = 'coral',
                 fill = False,
                 ax = axes[i])
        axes[i].set_xlabel(feat_disp[i])
        axes[0].set_ylabel('Density')
        
    plt.tight_layout()
    plt.savefig(namesave + '.png')
    plt.close()


def plot_heat_map(pis, alphas, betas, ys, num_seg, color_map, Tax = None, namesave = 'plot_heat_map', title = ''):

    """
    Plot a heatmap of predicted PDFs against true redshift values.

    This function visualizes the distribution of predicted redshifts (from Gamma mixture models)
    compared to the true redshifts, using a heatmap representation. The true redshift range is
    divided into segments, and for each segment, a representative PDF is plotted.

    Parameters
    ----------
    pis : ndarray of shape (n_samples, n_components)
        Mixture weights for each Gamma component per source.
    alphas : ndarray of shape (n_samples, n_components)
        Shape parameters (concentration) for each Gamma component per source.
    betas : ndarray of shape (n_samples, n_components)
        Rate parameters for each Gamma component per source.
    ys : ndarray of shape (n_samples,)
        True redshift values.
    num_seg : int
        Number of segments to divide the redshift range for visualization.
    color_map : str
        Name of the matplotlib colormap to use for the heatmap.
    Tax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure is created and the plot is saved.
    namesave : str, optional
        Base name for saving the plot as a PNG file if `Tax` is not provided. Default is 'plot_heat_map'.
    title : str, optional
        Title of the plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis object with the heatmap if `Tax` is provided.

    Notes
    -----
    This function samples one representative source from each redshift segment to compute
    the heatmap. If no sources fall within a segment, that column in the heatmap is set to zero.
    """

    count=0
    # Divide the range of ys into equal segments
    segment = (np.max(ys) - np.min(ys))/num_seg
    index_y = np.arange(len(ys)) # Indices of sources

    # Array to store the selected indices for each segment (-1 if no source in that segment)
    i_equi = -np.ones(num_seg)

    for i in range(num_seg):
        # Define lower and upper limits of the current segment
        z_l = count*segment + np.min(ys)
        count += 1
        z_u = count*segment + np.min(ys)

        # Find sources with ys within this segment
        i_trun = index_y[(ys>z_l).flatten() & (ys<z_u).flatten()]

        # Randomly select one source from this segment, if available
        if i_trun.size != 0:
            i_equi[i] = int(i_trun[np.random.randint(0, len(i_trun), 1)])

    i_equi = i_equi.astype(int) # Ensure integer indices

    # Initialize heatmap matrix
    heat = np.zeros((num_seg, num_seg))
    y_ph_z = np.linspace(0.0, np.max(ys), num_seg) # Grid for PDF computation

    for j in range(num_seg):
        if i_equi[j] != -1: # If a valid source exists for this segment
            final = np.zeros_like(y_ph_z)

            # Sum the weighted PDFs for this source
            for i, (weight_mix, alpha_mix, beta_mix) in enumerate(zip(pis[i_equi[j],:], alphas[i_equi[j],:], betas[i_equi[j],:])):
                dist = tfd.Gamma(concentration = alpha_mix, rate = beta_mix)
                pdf = dist.prob(y_ph_z)
                temp = pdf.numpy() * weight_mix
                final = final + temp

            # Normalize PDF and add to heatmap
            heat[:,j] = final/np.max(final)
        else:
            heat[:,j] = np.zeros_like(y_ph_z) # Empty segment

    # Create plot if axis not provided
    if Tax == None:
        fig, ax = plt.subplots()
    else:
        ax = Tax

    # Display heatmap
    ax.imshow(heat, cmap = color_map, origin = 'lower', extent = [0, np.max(ys), 0, np.max(ys)])
    ax.set_xticks(np.arange(0, 2, 0.5))
    ax.set_yticks(np.arange(0, 2, 0.5))
    ax.plot([0, np.max(ys)], [0, np.max(ys)], linewidth = 1, color = 'k')
    ax.set_xlabel('Redshift (Truth)', fontsize = 18)
    ax.set_ylabel('Prediction (Mean PDF)', fontsize = 18)
        
    # Save plot if axis was created internally
    if Tax == None:
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(namesave + '.png')
        plt.close()
    else:
        return ax


def mypairplot(df, columns, size=None, color = 'coral', namesave = 'mypairplot'):
    
    """
    Create and save a pairplot (scatterplot matrix) for selected features.

    This function visualizes pairwise relationships and univariate distributions 
    for a subset of features in a DataFrame. It optionally subsamples the data 
    to improve plotting performance on large datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset.
    columns : list of str
        List of column names to include in the pairplot.
    size : int, optional
        Number of samples to randomly select from the dataset. If None, the entire dataset is used.
    color : str, optional
        Color of the scatterplot points and KDE curves. Default is 'coral'.
    namesave : str, optional
        Base filename for saving the plot as a PNG file. Default is 'mypairplot'.

    Notes
    -----
    The plot is saved automatically to disk and is not returned.
    """    

    # Optionally sample the dataset to reduce plotting load
    if size is not None:
        df = df.sample(n=size)

    # Create pairplot with specified aesthetics
    pair_colors_scaled = sns.pairplot(df[columns], 
                                diag_kind = "kde", 
                                height = 1, 
                                aspect = 1.5, 
                                corner = True,
                                plot_kws = dict(s = 0.5, color = color, edgecolor = color),
                                diag_kws = dict(color = color))

    # Save the plot
    pair_colors_scaled.savefig(namesave + '.png')
    plt.close()


def pca_plot(X_pca, namesave = 'pca_plot', size = None, eigen_data = None):

    """
    Plot PCA projections in 2D planes (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3).

    This function creates scatterplots of the first three principal components (PC1, PC2, PC3)
    from PCA-transformed data. Optionally, it can overlay eigenvectors to visualize feature
    contributions to each principal component.

    Parameters
    ----------
    X_pca : numpy.ndarray
        Array of PCA-transformed data with shape (n_samples, n_components).
    namesave : str, optional
        Base filename to save the plot as a PNG file. Default is 'pca_plot'.
    size : int, optional
        Number of random samples to plot. If None, all data points are plotted.
    eigen_data : tuple, optional
        Tuple containing:
        - eigenvec : numpy.ndarray
            The matrix of eigenvectors (loadings) with shape (n_components, n_features).
        - df_data : pandas.DataFrame
            DataFrame containing the feature names (assumed to be in columns starting from the third position).

    Notes
    -----
    The plot consists of three subplots:
    - PC1 vs PC2
    - PC1 vs PC3
    - PC2 vs PC3

    If eigen_data is provided, the eigenvectors (feature loadings) are visualized as arrows
    on the scatterplots to indicate the contribution of each feature.
    """

    # Set sample size to total number of points if not specified
    if size == None:
        size = X_pca.shape[0]

    # Randomly sample data points to plot
    sample = np.random.randint(0, X_pca.shape[0], size)

    if eigen_data is not None:
        eigenvec = eigen_data[0]
        df_data = eigen_data[1]

    # Determine axis limits based on sampled data
    x_max, y_max, z_max = np.max(X_pca[sample, 0]), np.max(X_pca[sample, 1]), np.max(X_pca[sample, 2])
    x_min, y_min, z_min = np.min(X_pca[sample, 0]), np.min(X_pca[sample, 1]), np.min(X_pca[sample, 2])

    fig, ax = plt.subplots(1, 3, figsize = (18, 6))

    # Plot PC1 vs PC2
    ax[0].scatter(X_pca[sample, 0], X_pca[sample, 1], color = 'coral', s = 0.5)

    if eigen_data is not None:
        for i in range(X_pca.shape[1]):
            # Plot eigenvector arrows
            ax[0].arrow(0, 0, eigenvec[0, i]*0.7*x_max, eigenvec[1,i]*0.7*y_max, 
                    color = 'tab:blue', alpha = 0.5, width = 0.01, head_width = 0.1)
            ax[0].text(eigenvec[0, i]*x_max, eigenvec[1, i]*y_max,
                list(df_data)[i + 2], color = 'tab:blue')
    
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')

    # Plot PC1 vs PC3
    ax[1].scatter(X_pca[sample, 0], X_pca[sample, 2], color = 'coral', s = 0.5)

    if eigen_data is not None:
        for i in range(X_pca.shape[1]):
            ax[1].arrow(0, 0, eigenvec[0, i]*0.7*x_max,eigenvec[2, i]*0.7*z_max, 
                    color = 'tab:blue', alpha = 0.5, width = 0.01, head_width = 0.1)
            ax[1].text(eigenvec[0,i]*x_max, eigenvec[2,i]*z_max,
                list(df_data)[i+2], color = 'tab:blue')

    ax[1].set_xlabel('PC1')
    ax[1].set_ylabel('PC3')

    # Plot PC2 vs PC3
    ax[2].scatter(X_pca[sample, 1], X_pca[sample, 2], color = 'coral', s = 0.5)

    if eigen_data is not None:
        for i in range(X_pca.shape[1]):
            ax[2].arrow(0, 0, eigenvec[1, i]*0.7*y_max,eigenvec[2, i]*0.7*z_max, 
                    color = 'tab:blue', alpha = 0.5, width = 0.01, head_width = 0.1)
            ax[2].text(eigenvec[1, i]*y_max, eigenvec[2, i]*z_max,
                list(df_data)[i + 2], color = 'tab:blue')

    ax[2].set_xlabel('PC2')
    ax[2].set_ylabel('PC3')

    # Set symmetric axis limits for visual consistency
    lim_x = max(np.abs(x_min), x_max)
    lim_y = max(np.abs(y_min), y_max)
    lim_z = max(np.abs(z_min), z_max)

    ax[0].set_xlim(-lim_x, lim_x)
    ax[0].set_ylim(-lim_y, lim_y)

    ax[1].set_xlim(-lim_x, lim_x)
    ax[1].set_ylim(-lim_z, lim_z)

    ax[2].set_xlim(-lim_y, lim_y)
    ax[2].set_ylim(-lim_z, lim_z)

    plt.tight_layout()
    plt.savefig(namesave + '.png')
    plt.close()


def pca_var(df_pc, namesave = 'pca_var', title = ''):

    """
    Plot explained variance and cumulative explained variance from PCA.

    This function generates a dual-axis plot showing both the explained variance
    and cumulative explained variance for each principal component.

    Parameters
    ----------
    df_pc : pandas.DataFrame
        A dataframe containing at least the following columns:
        - 'x' : Principal component index.
        - 'Var' : Explained variance (%) for each component.
        - 'Cum_var' : Cumulative explained variance (%).
    namesave : str, optional
        Base filename for saving the plot as a PNG file. Default is 'pca_var'.
    title : str, optional
        Title of the plot.

    Notes
    -----
    The plot is saved automatically to disk and is not returned.
    """

    fig, ax = plt.subplots(figsize = (7, 5))

    # Create a secondary Y-axis to plot cumulative variance
    ax2 = ax.twinx()

    # Plot explained variance
    sns.lineplot(x = "x", y = "Var",
              marker = "o",
             color = "coral",
             data = df_pc, ax = ax)

    # Plot cumulative explained variance
    sns.lineplot(x = "x", y = "Cum_var",
              marker = "o",
             color = 'black',
             data = df_pc, ax = ax2)

    # Customize axis labels and title
    ax2.set(xlabel = 'Principal Component', ylabel = 'Explained Cumulative \n Variance (%)')
    ax.set_ylabel(ylabel = 'Explained Variance (%)', color = 'coral', fontsize = 15)  # we already handled the x-label with ax1
    ax.set_xlabel(xlabel = 'Principal Component', fontsize = 15)  # we already handled the x-label with ax1
    ax.set_title(title, style = 'italic', fontsize = 14)

    # Adjust axis label colors and tick sizes
    ax.tick_params(axis = 'y', labelcolor = 'coral', labelsize = 13)
    ax2.set_ylabel(ylabel = 'Accumulated Explained \n Variance (%)', color = 'black', fontsize = 15)  # we already handled the x-label with ax1
    ax2.tick_params(axis = 'y', labelsize = 13)

    # Save the figure
    plt.tight_layout()
    plt.savefig(namesave + '.png')
    plt.close()


def train_val_loss(epochs_train, loss_train, ri_train, namesave = 'train_val_loss', title = '', valdata = None):
    
    """
    Plot training and validation loss and reliability index (RI) across epochs.

    This function generates two plots:
    - Training and optional validation loss over epochs.
    - Training and optional validation reliability index (RI) over epochs.

    Parameters
    ----------
    epochs_train : array-like
        List or array of training epochs.
    loss_train : array-like
        Training loss values for each epoch.
    ri_train : array-like
        Training reliability index (RI) values for each epoch.
    namesave : str, optional
        Base filename to save the plot as a PNG file. Default is 'train_val_loss'.
    title : str, optional
        Title suffix for the plot, displayed in italics.
    valdata : tuple, optional
        Tuple containing:
        - val_epochs : array-like
            Validation epochs.
        - val_loss : array-like
            Validation loss values.
        - val_ri : array-like
            Validation RI values.

    Notes
    -----
    Vertical lines are plotted at the epochs where validation loss and RI reach their minimum.
    """

    fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (15, 3.5))

    color_train = "coral"
    color_val = "tab:blue"
    legend_labels = ['Training']

    # Plot training loss
    ax[2].plot(epochs_train, loss_train, color = color_train)

    # Plot training reliability index (RI)
    ax[3].plot(epochs_train, ri_train, color = color_train)

    if valdata is not None:
        val_epochs, val_loss, val_ri = valdata

        # Epochs where validation loss and RI are minimal
        epoch_min_val_loss = np.argmin(val_loss) + 1
        epoch_min_val_ri = np.argmin(val_ri) + 1

        # Plot validation loss
        ax[2].plot(val_epochs, val_loss, color = color_val)
        ax[2].axvline(x = epoch_min_val_loss, color = 'black', alpha = 1)
        legend_labels.append('Validation')

        # Plot validation RI
        ax[3].plot(val_epochs, val_ri, color=color_val)
        ax[3].axvline(x = epoch_min_val_ri, color = 'black', alpha = 1)

    # Loss subplot formatting
    ax[2].set_title(r'$\gamma$-MDN %s' % title, style = 'italic', fontsize = 14)
    ax[2].set_ylabel('Loss', fontsize = 15)
    ax[2].set_xlabel('Epoch', fontsize = 15)
    ax[2].legend(legend_labels, loc = 'upper right', fontsize = 13)
    ax[2].grid()

    # RI subplot formatting
    ax[3].set_title(r'$\gamma$-MDN %s' % title, style = 'italic', fontsize = 14)
    ax[3].set_ylabel('RI', fontsize = 15)
    ax[3].set_xlabel('Epoch', fontsize = 15)
    ax[3].legend(legend_labels, loc='upper right', fontsize = 13)
    ax[3].grid()

    plt.tight_layout()
    plt.savefig(namesave + '.png')
    plt.close()


def pit_plot(pit_train, ri_train, pit_test = None, ri_test = None, namesave = 'pit_plot', title = '',
             n_bins_train = 20, n_bins_test = 20):

    """
    Plot Probability Integral Transform (PIT) histograms for training and optional test data.

    Parameters
    ----------
    pit_train : torch.Tensor or np.ndarray
        PIT values for the training set.
    ri_train : float
        Reliability Index (RI) for the training set.
    pit_test : torch.Tensor or np.ndarray, optional
        PIT values for the test set.
    ri_test : float, optional
        Reliability Index (RI) for the test set.
    namesave : str, optional
        Base filename to save the plot as a PNG file. Default is 'pit_plot'.
    title : str, optional
        Plot title.
    n_bins_train : int, optional
        Number of histogram bins for the training PIT plot.
    n_bins_test : int, optional
        Number of histogram bins for the test PIT plot.

    Notes
    -----
    A perfectly calibrated PIT histogram should be uniform across bins.
    """

    if hasattr(pit_train, 'numpy'):
        pit_train = pit_train.numpy()
    if pit_test is not None and hasattr(pit_test, 'numpy'):
        pit_test = pit_test.numpy()

    if pit_test is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize = (5, 4))

    # Plot training PIT
    sns.histplot(pit_train, bins = n_bins_train, stat = "density", color = 'orange', legend = False, ax = ax1)

    # Reference lines for perfect calibration and 90% CI
    ax1.axhline(y = 1., linestyle = '--', color = 'black', alpha = 0.7)
    ax1.axhline(y = 0.95, linestyle = '-.', color = 'black', alpha = 0.7)
    ax1.axhline(y = 1.05, linestyle = '-.', color = 'black', alpha = 0.7)
    ax1.axvline(x = 0.5, linestyle = '--', color = 'black', alpha = 0.7)

    ax1.set_ylim(bottom = 0)
    ax1.set_xlabel('PIT', fontsize = 15)
    ax1.set_ylabel('Relative Frequency', fontsize = 15)
    ax1.set_title('Train (RI = %.4f)' % ri_train, style = 'italic', fontsize = 14)
    ax1.tick_params(axis = 'both', labelsize = 13)

    # Plot test PIT if provided
    if pit_test is not None:
        sns.histplot(pit_test, bins = n_bins_test, stat = "density", color = 'tab:blue', legend = False, ax = ax2)

        ax2.axhline(y = 1., linestyle = '--', color = 'black', alpha = 0.7)
        ax2.axhline(y = 0.95, linestyle = '-.', color = 'black', alpha = 0.7)
        ax2.axhline(y = 1.05, linestyle = '-.', color = 'black', alpha = 0.7)
        ax2.axvline(x = 0.5, linestyle = '--', color = 'black', alpha = 0.7)

        ax2.set_ylim(bottom = 0)
        ax2.set_xlabel('PIT', fontsize = 15)
        ax2.set_title('Test (RI = %.4f)' % ri_test, style = 'italic', fontsize = 14)
        ax2.tick_params(axis = 'both', labelsize = 13)

    plt.suptitle(title, fontsize = 14, style = 'italic')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(namesave + '.png')
    plt.close()


def dens_hist(var, label = '', namesave = 'dens_hist', title = ''):

    """
    Plot a density-normalized histogram.

    Parameters
    ----------
    var : array-like
        The data to plot.
    label : str, optional
        Label for the x-axis.
    namesave : str, optional
        Base filename to save the plot as a PNG file.
    title : str, optional
        Title for the plot.
    """

    fig, ax = plt.subplots(figsize = (5, 5))
    ax.hist(var, density = True, color = 'coral', edgecolor = 'black', alpha = 0.7)

    ax.set_ylabel('Density', fontsize = 14)
    ax.set_xlabel(label, fontsize = 14)
    ax.set_title(title, fontsize = 15)
    ax.tick_params(axis = 'both', labelsize = 12)

    plt.tight_layout()
    plt.savefig(namesave + '.png')
    plt.close()


def plot_examples_pdfs(Y, pi, alpha, beta, num_cols = 6, num_rows = 3, namesave = 'plot_examples_pdfs', title = '', zmax = 2., ordered = None):

    """
    Plot example PDFs for multiple test sources in a grid.

    Parameters
    ----------
    Y : array-like
        True redshift values.
    pi : array-like
        Mixture weights for each source.
    alpha : array-like
        Alpha parameters of the Gamma components.
    beta : array-like
        Beta parameters of the Gamma components.
    num_cols : int, optional
        Number of columns in the plot grid.
    num_rows : int, optional
        Number of rows in the plot grid.
    namesave : str, optional
        Base filename to save the plot.
    title : str, optional
        Plot title.
    zmax : float, optional
        Maximum redshift value on x-axis.
    ordered : array-like, optional
        Optional ordered indices to select specific sources.

    Notes
    -----
    The function assumes the existence of a 'plot_gamma_mix' function
    that plots a gamma mixture model on the given axis.
    """
    
    fig, axes = plt.subplots(nrows = num_rows, ncols = num_cols, figsize=(3.4 * num_cols, 3.4 * num_rows))
    axes = axes.flatten()

    num_gal = num_cols * num_rows

    if ordered is None:
        gal_id = np.random.choice(len(Y), num_gal, replace = False)
    else:
        gal_id = ordered[-num_gal:]

    for count, ax in enumerate(axes):
        plot_gamma_mix(pi[gal_id[count]], alpha[gal_id[count]], beta[gal_id[count]], ax = ax, color = 'coral')
        ax.axvline(x = Y[gal_id[count]], color = 'black', alpha = 0.7)
        ax.set_xlim(0, zmax)
        ax.set_ylabel('PDF', fontsize = 14)
        ax.set_xlabel('Redshift', fontsize = 14)
        ax.set_title(f'Test source {gal_id[count]}', fontsize = 14)
        ax.tick_params(axis = 'both', labelsize = 12)

    axes[0].legend(['Prediction', 'Truth'], loc = 'best', fontsize = 12)

    plt.suptitle(title, fontsize = 16, style = 'italic')
    plt.tight_layout(rect = [0, 0, 1, 0.95])
    plt.savefig(namesave + '.png')
    plt.close()


def plot_mean_true(gamma_pdf, Y, gamma_pdf_train = None, Y_train = None,
                   namesave = 'plot_mean_true', title = '', other = None):

    """
    Plot predicted mean vs true redshift for train and test sets.

    Parameters
    ----------
    gamma_pdf : dict
        Dictionary containing at least the key 'Mean' for predicted means (test set).
    Y : array-like
        True redshift values (test set).
    gamma_pdf_train : dict, optional
        Dictionary containing predicted means for the training set.
    Y_train : array-like, optional
        True redshift values (training set).
    namesave : str, optional
        Filename to save the plot.
    title : str, optional
        Overall plot title.
    other : str, optional
        Optional additional key from gamma_pdf to plot as a comparison.
    """

    ymax = np.max(Y)

    if gamma_pdf_train is not None:
        ymax_train = np.max(Y_train)
        ymax = max(ymax, ymax_train)

        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6), sharey = True)

        # Plot for training set
        ax_train = axes[0]
        ax_test = axes[1]

        ax_train.scatter(Y_train, gamma_pdf_train['Mean'], alpha = 0.5, s = 6, c = 'coral')
        ax_train.plot([0, ymax], [0, ymax], linewidth = 1, color = 'k')
        ax_train.set_xlim(0, ymax)
        ax_train.set_ylim(0, ymax)
        ax_train.set_xlabel('Redshift (Truth)', fontsize = 16)
        ax_train.set_ylabel('Prediction (Mean PDF)', fontsize = 16)
        ax_train.set_title('Train', fontsize = 15)
        ax_train.tick_params(axis='both', labelsize = 12)

        ax = ax_test
        ax.set_title('Test', fontsize = 15)

    else:
        fig, ax = plt.subplots(figsize = (6, 6))

    # Plot for test set
    ax.scatter(Y, gamma_pdf['Mean'], alpha = 0.5, s = 6, c = 'coral', label = 'GMDNz')

    if other is not None:
        ax.scatter(Y, gamma_pdf[other], alpha = 0.5, s = 6, c = 'blue', label = other)
        ax.legend(fontsize=12)

    ax.plot([0, ymax], [0, ymax], linewidth = 1, color = 'k')
    ax.set_xlim(0, ymax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Redshift (Truth)', fontsize = 16)

    if gamma_pdf_train is None:
        ax.set_ylabel('Prediction (Mean PDF)', fontsize = 16)

    ax.tick_params(axis = 'both', labelsize = 12)

    plt.suptitle(title, fontsize = 17)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(namesave + '.png')
    plt.close()


def pit_plot_z(output, z_pred, Y, redshift_bins = None, n_bins = 20, namesave = 'pit_plot_z', title = ''):

    """
    Plot PIT histograms in redshift bins and compute RI in each bin.

    Parameters
    ----------
    output : np.array
        Predicted distributions (samples or density arrays) for each object.
    z_pred : np.array
        Predicted redshift values (e.g., mean or mode of the predicted distributions).
    Y : np.array
        True redshift values.
    redshift_bins : np.array, optional
        Bin edges for redshift binning. If None, defaults to np.arange(0.1, 1.4, 0.1).
    n_bins : int, optional
        Number of bins for PIT histograms.
    namesave : str, optional
        Filename to save the plot.
    title : str, optional
        Plot title.

    Returns
    -------
    z_rel : np.array
        Midpoints of each redshift bin.
    RI : np.array
        RI metric in each redshift bin.
    """

    if redshift_bins is None:
        redshift_bins = np.arange(0.1, 1.4, 0.1)

    dimension = len(redshift_bins) - 1
    z_rel = np.zeros(dimension)
    RI = np.zeros(dimension)

    fig, ax = plt.subplots(1, dimension, figsize = (4 * dimension, 4))

    if dimension == 1:
        ax = [ax]  # Ensure ax is always iterable

    for i in range(dimension):
        z_rel[i] = (redshift_bins[i] + redshift_bins[i + 1]) / 2.

        zmask = (z_pred >= redshift_bins[i]) & (z_pred < redshift_bins[i + 1])

        temp_output = output[zmask, :]
        temp_y = Y[zmask]

        if len(temp_y) == 0:
            print(f'Warning: No data in bin {i} [{redshift_bins[i]}, {redshift_bins[i+1]}]')
            continue

        RI[i] = GM.RI_metric(temp_y[:, np.newaxis], temp_output).numpy()
        PIT = GM.PIT(temp_y[:, np.newaxis], temp_output).numpy()

        sns.histplot(PIT,
                     stat = 'density',
                     bins = n_bins,
                     color = 'tab:blue',
                     fill = False,
                     legend = False,
                     ax = ax[i])

        ax[i].set_title(f'z_rel = {z_rel[i]:.2f}', fontsize = 12)
        ax[i].axhline(y = 1.0, linestyle = '--', color = 'black', alpha = 0.7)
        ax[i].axvline(x = 0.5, linestyle = '--', color = 'black', alpha = 0.7)
        ax[i].set_ylim(bottom = 0)
        ax[i].set_xlabel('PIT')
        ax[i].set_ylabel('Density')

    plt.suptitle(title, fontsize = 15)
    plt.tight_layout(rect = [0, 0, 1, 0.95])
    plt.savefig(namesave + '.png')
    plt.close()

    return z_rel, RI


def val_with_z(val, z, valname = 'value', namesave = 'val_with_z', title = ''):

    """
    Plot a given metric as a function of redshift.

    Parameters
    ----------
    val : np.array
        Metric values to plot (e.g., RI per redshift bin).
    z : np.array
        Redshift bin centers.
    valname : str, optional
        Label for the plotted metric (y-axis label).
    namesave : str, optional
        Base filename for saving the plot.
    title : str, optional
        Plot title.
    """

    plt.figure(figsize = (6, 4))
    plt.plot(z, val, c = 'tab:blue', marker = 'o', linestyle = '-')
    plt.ylabel(valname, fontsize = 14)
    plt.xlabel('Redshift', fontsize = 14)
    plt.title(title, fontsize = 15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{namesave}_{valname}.png')
    plt.close()

