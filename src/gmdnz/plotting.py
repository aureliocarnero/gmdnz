import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import pandas as pd

import gamma_mdn as GM

def plot_gamma_mix(pis, alphas, betas, ax = None, color = 'red', maxredshift = 5., bins = 600, namesave = 'plot_gamma_mix'):
    """
    Plots the mixture of gamma models to axis=ax
    """
    x = np.linspace(0.0, maxredshift, bins)
    final = np.zeros_like(x)
    for i, (weight_mix, alpha_mix, beta_mix) in enumerate(zip(pis, alphas, betas)):
        dist = tfd.Gamma(concentration = alpha_mix, rate = beta_mix)
        pdf = dist.prob(x)
        temp = pdf.numpy() * weight_mix
        final = final + temp
    if ax == None:
        fig, ax = plt.subplots()
    ax.plot(x, final, c = color)

    if ax == None:
        plt.savefig(namesave + '.png')
        plt.close()
    else:
        return ax


# function to plot variables
def plot_variables(data, feat_disp, namesave = 'plot_variables'):
    fig, axes = plt.subplots(nrows = 1, ncols = len(feat_disp), figsize = (20,3))
    for i in range(int(len(feat_disp))):
    # Density Plot and Histogram of all arrival delays
        sns.histplot(data[feat_disp[i]],
                 element = "step",
                 stat = "density",
                 bins = int(180/5),
                 color = 'coral',
                 fill = False,
                 ax = axes[i])
        axes[i].set_xlabel(feat_disp[i])
        axes[0].set_ylabel('Density')
        #axes[i].set_yscale('log')
    plt.tight_layout()
    plt.savefig(namesave + '.png')
    plt.close()

def plot_heat_map(pis, alphas, betas, ys, num_seg, color_map, Tax = None, namesave = 'plot_heat_map', title = ''):
    count=0
    # devide Y values in num_seg segments
    segment = (np.max(ys) - np.min(ys))/num_seg
    # store index of non ordered Ys
    index_y = np.arange(len(ys))
    # array to store indeces of "equispaced" Y values, negative if no value
    i_equi = -np.ones(num_seg)
    for i in range(num_seg):
        # select lower and upper limits of i segment
        z_l = count*segment + np.min(ys)
        count += 1
        z_u = count*segment + np.min(ys)
        # select indeces of sources with Y values between z_l and z_u
        i_trun = index_y[(ys>z_l).flatten() & (ys<z_u).flatten()]
        # if this array is not empty, select one of those randomly
        if i_trun.size != 0:
            i_equi[i] = int(i_trun[np.random.randint(0, len(i_trun), 1)])
    i_equi = i_equi.astype(int)

    # initialize heat-map
    heat = np.zeros((num_seg, num_seg))
    # For each of those sources, plot the PDF
    y_ph_z = np.linspace(0.0, np.max(ys), num_seg)
    for j in range(num_seg):
        if i_equi[j] != -1: # if the array is not empty
            final = np.zeros_like(y_ph_z)
            for i, (weight_mix, alpha_mix, beta_mix) in enumerate(zip(pis[i_equi[j],:], alphas[i_equi[j],:], betas[i_equi[j],:])):
                dist = tfd.Gamma(concentration = alpha_mix, rate = beta_mix)
                pdf = dist.prob(y_ph_z)
                temp = pdf.numpy() * weight_mix
                final = final + temp
            heat[:,j] = final/np.max(final)
        else:
            heat[:,j] = np.zeros_like(y_ph_z)
    if Tax == None:
        fig, ax = plt.subplots()
    ax.imshow(heat, cmap = color_map, origin = 'lower', extent = [0, np.max(ys), 0, np.max(ys)])
    ax.set_xticks(np.arange(0, 2, 0.5))
    ax.set_yticks(np.arange(0, 2, 0.5))
    ax.plot([0, np.max(ys)], [0, np.max(ys)], linewidth = 1, color = 'k')
    ax.set_xlabel('Redshift (Truth)', fontsize = 18)
    ax.set_ylabel('Prediction (Mean PDF)', fontsize = 18)
    
    if Tax == None:
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(namesave + '.png')
        plt.close()
    else:
        return ax

def mypairplot(df, columns, size=None, color = 'coral', namesave = 'mypairplot'):
    
    if size is not None:
        df = df.sample(n=size)

    pair_colors_scaled = sns.pairplot(df[columns], diag_kind = "kde", height = 1, aspect = 1.5, corner = True,
                                  plot_kws = dict(s = 0.5, color = color, edgecolor = color),
                                  diag_kws = dict(color = color))

    pair_colors_scaled.savefig(namesave + '.png')
    plt.close()


def pca_plot(X_pca, namesave = 'pca_plot', size = None, eigen_data = None):



    if size == None:
        size = X_pca.shape[0]

    sample = np.random.randint(0, X_pca.shape[0], size)

    if eigen_data is not None:
        eigenvec = eigen_data[0]
        df_data = eigen_data[1]

    x_max = np.max(X_pca[sample,0])
    y_max = np.max(X_pca[sample,1])
    z_max = np.max(X_pca[sample,2])
    x_min = np.min(X_pca[sample,0])
    y_min = np.min(X_pca[sample,1])
    z_min = np.min(X_pca[sample,2])

    

    fig, ax = plt.subplots(1, 3, figsize = (18, 6))

    ax[0].scatter(X_pca[sample, 0], X_pca[sample, 1], color = 'coral', s = 0.5)

    if eigen_data is not None:
        for i in range(X_pca.shape[1]):
            ax[0].arrow(0, 0, eigenvec[0, i]*0.7*x_max, eigenvec[1,i]*0.7*y_max, color = 'tab:blue', alpha = 0.5, width = 0.01, head_width = 0.1)
            ax[0].text(eigenvec[0, i]*x_max, eigenvec[1, i]*y_max,
                list(df_data)[i + 2], color = 'tab:blue')
    
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')


    ax[1].scatter(X_pca[sample, 0], X_pca[sample, 2], color = 'coral', s = 0.5)

    if eigen_data is not None:
        for i in range(X_pca.shape[1]):
            ax[1].arrow(0, 0, eigenvec[0, i]*0.7*x_max,eigenvec[2, i]*0.7*z_max, color = 'tab:blue', alpha = 0.5, width = 0.01, head_width = 0.1)
            ax[1].text(eigenvec[0,i]*x_max, eigenvec[2,i]*z_max,
                list(df_data)[i+2], color = 'tab:blue')

    ax[1].set_xlabel('PC1')
    ax[1].set_ylabel('PC3')

    ax[2].scatter(X_pca[sample, 1], X_pca[sample, 2], color = 'coral', s = 0.5)

    if eigen_data is not None:
        for i in range(X_pca.shape[1]):
            ax[2].arrow(0, 0, eigenvec[1, i]*0.7*y_max,eigenvec[2, i]*0.7*z_max, color = 'tab:blue', alpha = 0.5, width = 0.01, head_width = 0.1)
            ax[2].text(eigenvec[1, i]*y_max, eigenvec[2, i]*z_max,
                list(df_data)[i + 2], color = 'tab:blue')
    ax[2].set_xlabel('PC2')
    ax[2].set_ylabel('PC3')

    # set axis limits
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
    fig, ax = plt.subplots(figsize = (7, 5))

    # With Y channel
    ax2 = ax.twinx() #This allows the common axes (flow rate) to be shared
    sns.lineplot(x = "x", y = "Var",
              marker = "o",
             color = "coral",
             data = df_pc, ax = ax)

    sns.lineplot(x = "x", y = "Cum_var",
              marker = "o",
             color = 'black',
             data = df_pc, ax = ax2)

    #ax.set(xlabel='Principal Component', ylabel='Explained Variance (%)')
    ax2.set(xlabel = 'Principal Component', ylabel = 'Explained Cumulative \n Variance (%)')
    ax.set_ylabel(ylabel = 'Explained Variance (%)', color = 'coral', fontsize = 15)  # we already handled the x-label with ax1
    ax.set_xlabel(xlabel = 'Principal Component', fontsize = 15)  # we already handled the x-label with ax1
    ax.set_title(title, style = 'italic', fontsize = 14)
    ax.tick_params(axis = 'y', labelcolor = 'coral', labelsize = 13)
#    ax.xaxis.set_ticks(np.arange(0, X_pca_w.shape[1] + 2, 2))
    ax2.set_ylabel(ylabel = 'Accumulated Explained \n Variance (%)', color = 'black', fontsize = 15)  # we already handled the x-label with ax1
    ax2.tick_params(axis = 'y', labelsize = 13)

    plt.tight_layout()
    plt.savefig(namesave + '.png')
    plt.close()

#Plot training & validation loss values
def train_val_loss(epochs_train, loss_train, ri_train, namesave = 'train_val_loss', title = '', valdata = None):
    fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (15, 3.5))

    # plot for crps
    # losses
    color_1 = "coral"
    color_2 = "tab:blue"
    leg = ['Training']
    # plot for LL
    ax[2].plot(epochs_train, loss_train, c = color_1)
    ax[3].plot(epochs_train, ri_train, c = color_1)

    if valdata is not None:
        epoch_min_val_loss = np.argmin(valdata[1]) + 1
        epoch_min_val_ri = np.argmin(valdata[2]) + 1

        ax[2].plot(valdata[0], valdata[1], c = color_2)

        ax[2].axvline(x = epoch_min_val_loss, color = 'black', alpha = 1)
        leg.append('Validation')

        ax[3].plot(valdata[0], valdata[2], c = color_2)

        x[3].axvline(x = epoch_min_val_ri, color = 'black', alpha = 1)

    ax[2].set_title('$\gamma$-MDN %s' % title, style = 'italic', fontsize = 14)

    ax[2].set_ylabel('Loss', fontsize = 15)
    ax[2].set_xlabel('Epoch', fontsize = 15)

    ax[2].legend(leg, loc = 'upper right', fontsize = 13)
    ax[2].grid()


    x[3].set_title('$\gamma$-MDN %s' % title, style = 'italic', fontsize = 14)

    x[3].set_ylabel('RI', style = 'italic', fontsize = 15)
    x[3].set_xlabel('Epoch', fontsize = 15)
    x[3].legend(leg, loc = 'upper right', fontsize = 13)
    x[3].grid()


    plt.tight_layout()
    plt.savefig(namesave + '.png')
    plt.close()

def pit_plot(pit_train, ri_train, pit_test = None, ri_test = None, namesave = 'pit_plot', title = '', n_bins_train = 20, n_bins_test = 20):

    if pit_test is not None:
        figsize = (8, 4)
        name = 'Train'
    else:
        figsize = (4, 4)
        name = ''

    fig = plt.figure(figsize = figsize)

    if pit_test is not None:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    else:
        ax1 = fig.add_subplot(111)



    sns.histplot(pit_train.numpy(),
                 bins = n_bins_train,
                 stat = "density",
                 color = 'orange',
                 legend = False,
                 ax = ax1)

    ax1.axhline(y = 1., linestyle = '--', color = 'black', alpha = 0.7)
    ax1.axhline(y = 0.95, linestyle = '-.', color = 'black', alpha = 0.7)
    ax1.axhline(y = 1.05, linestyle = '-.', color = 'black', alpha = 0.7)
    ax1.axvline(x = 0.5, linestyle = '--', color = 'black', alpha = 0.7)

    ax1.set_ylim(bottom = 0)
    ax1.set_xlabel('PIT', fontsize = 15)
    ax1.set_ylabel('Relative Frequency', fontsize = 15)

    ax1.set_title('%s ($RI$ = %.4f)' %(name, ri_train), style = 'italic', fontsize = 14)
    ax1.tick_params(axis = 'y', labelsize = 13)
    ax1.tick_params(axis = 'x', labelsize = 13)

    if pit_test is not None:
        sns.histplot(pit_test.numpy(),
                     bins = n_bins_test,
                     stat = "density",
                     color = 'tab:blue',
                     legend = False,
                     ax = ax2)

        # Perfectly callibrated fit together with 90% CI\n",
        ax2.axhline(y = 1., linestyle = '--', color = 'black', alpha = 0.7)
        ax2.axhline(y = 0.95, linestyle = '-.', color = 'black', alpha = 0.7)
        ax2.axhline(y = 1.05, linestyle = '-.', color = 'black', alpha = 0.7)
        ax2.axvline(x = 0.5, linestyle = '--', color = 'black', alpha = 0.7)

        ax2.set_ylim(bottom = 0)

        ax2.set_xlabel('PIT', fontsize = 15)
        #ax2.set_ylabel('Relative Frequency', fontsize = 15)

        ax2.set_title('Test ($RI$ = %.4f)' %(ri_test), style = 'italic', fontsize = 14)

        ax2.tick_params(axis = 'y', labelsize = 13)
        ax2.tick_params(axis = 'x', labelsize = 13)

#    height = 0.97
#    fig.text(0.01, height, "a)", horizontalalignment = 'left', verticalalignment = 'center', fontsize = 15)
#    fig.text(0.505, height, "b)", horizontalalignment = 'left', verticalalignment = 'center', fontsize = 15)

    #fig.text(0.4, height, "$\gamma$-MDN %s" % title, horizontalalignment = 'left', verticalalignment = 'center', fontsize = 14, style = 'italic')

    plt.tight_layout()
    plt.suptitle(title)
    plt.savefig(namesave + '.png')
    plt.close()

def dens_hist(var, label = '', namesave = 'dens_hist', title = ''):
    fig, ax = plt.subplots(figsize = (5, 5))
    ax.hist(var, density = True)
    ax.set_ylabel('Frequency')
    ax.set_xlabel(label)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(namesave + '.png')
    plt.close()


def plot_examples_pdfs(Y, pi, alpha, beta, num_cols = 6, num_rows = 3, namesave = 'plot_examples_pdfs', title = '', zmax = 1.5, ordered = None):
    
    fig, axes = plt.subplots(nrows = num_rows, ncols = num_cols, figsize=(3.4 * num_cols, 3.4 * num_rows))
    
    num_gal = num_cols * num_rows

    if ordered is None:
        gal_id = np.random.randint(1, len(Y), num_gal)

    else:
        gal_id = ordered[-num_gal:]

    count = 0
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i, j] = plot_gamma_mix(pi[gal_id][count], alpha[gal_id][count], beta[gal_id][count], ax = axes[i, j], color = 'coral')
            axes[i, j].axvline(x = Y[gal_id[count]], color = 'black', alpha = 0.7)
            axes[i, j].set_xlim(0, zmax)
            axes[i, j].set_ylabel('PDF', fontsize = 14)
            axes[i, j].set_title('Test source ' + str(gal_id[count]), fontsize = 14)
            axes[i, j].set_xlabel('Redshift', fontsize = 14)
            count += 1

    axes[0, 0].legend(['Prediction', 'Truth'], loc = 'best', fontsize = 12)
    plt.suptitle(title)
    fig.tight_layout()
    plt.savefig(namesave + '.png')
    plt.close()


def plot_mean_true(gamma_pdf, Y, gamma_pdf_train = None, Y_train = None, namesave = 'plot_mean_true', title = '', other = None):
    ymax = np.max(Y)

    if gamma_pdf_train is not None:
        ymax_train = np.max(Y_train)
        ymax = np.max([ymax, ymax_train])
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5), sharey = True)
        ax = axes[1]
        ax_aux = axes[0]
        ax_aux.scatter(Y_train, gamma_pdf_train['Mean'], alpha = 0.5, s = 4, c = 'coral')
        ax_aux.plot([0, ymax], [0, ymax], linewidth = 1, color = 'k')
        ax_aux.set_xlim(0, ymax)
        ax_aux.set_ylim(0, ymax)
        ax_aux.set_xlabel('Redshift (Truth)', fontsize = 18)
        ax_aux.set_ylabel('Prediction (Mean PDF)', fontsize = 18)
        ax_aux.set_title('Train')
        ax.set_title('Test')

    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(Y, gamma_pdf['Mean'], alpha = 0.5, s = 4, c = 'coral', label = 'GMDNz')
    if other is not None:
        ax.scatter(Y, gamma_pdf[other], alpha = 0.5, s = 4, c = 'blue', label = other)
        ax.legend()
    ax.plot([0, ymax], [0, ymax], linewidth = 1, color = 'k')
    ax.set_xlim(0, ymax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Redshift (Truth)', fontsize = 18)
    

    if gamma_pdf_train is None:
        ax.set_ylabel('Prediction (Mean PDF)', fontsize = 18)


    plt.suptitle(title)
    fig.tight_layout()
    plt.savefig(namesave + '.png')
    plt.close()



def pit_plot_z(output, z_pred, Y, redshift_bins = None, n_bins = 20, namesave = 'pit_plot_z', title = ''):

    if redshift_bins is None:

        redshift_bins = np.arange(0.1, 1.4, 0.1)


    dimension = len(redshift_bins) - 1

    z_rel = np.zeros(dimension)

    RI = np.zeros(dimension)

    fig, ax = plt.subplots(1, dimension, figsize = (25, 4))


    for i1 in range(dimension):

        z_rel[i1] = (redshift_bins[i1] + redshift_bins[i1 + 1]) / 2.

        zmask = (z_pred >= redshift_bins[i1]) & (z_pred < redshift_bins[i1 + 1])
        
        temp_output = output[zmask, :]

        temp_y = Y[zmask]

        RI[i1] = GM.RI_metric(temp_y[:, np.newaxis], temp_output).numpy()

        PIT = GM.PIT(temp_y[:, np.newaxis], temp_output).numpy()

        n_bins = 20
      
        sns.histplot(PIT,
            stat = 'density',
            bins = n_bins, color = 'tag:blue',
            fill = False,
            legend = False,
            ax=ax[i1])

        ax[i1].set_title('z_rel = %.2f' % (z_rel[i1]), fontsize = 10)

        ax[i1].axhline(y = 1., linestyle = '--', color = 'black', alpha = 0.5)
        ax[i1].axvline(x = 0.5, linestyle = '--', color = 'black', alpha = 0.5)


    plt.tight_layout()
    plt.suptitle(title)
    plt.savefig(namesave + '.png')
    plt.close()

    return z_rel, RI

def val_with_z(val, z, valname = 'value', namesave = 'pit_plot_z', title = ''):
    plt.plot(z, val, c = 'blue')
    plt.ylabel(valname)
    plt.xlabel('Redshift')
    plt.title(title)
    plt.savefig(namesave + '_' + valname + '.png')
    plt.close()

    

