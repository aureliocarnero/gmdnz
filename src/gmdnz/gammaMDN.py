import numpy as np
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Lambda
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

from tensorflow.random import set_seed
set_seed(314)
#!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd
from astropy.table import Table

import tensorflow as tf
import tensorflow.math as tfm
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def get_mixture_coef(output):
    """
    Mapping output layer to mixute components and shape
    parameters of the distribution (pi, alpha, beta)
    """    
    ### split the last layer in 3 'blocks'
    out_pi, out_alpha, out_beta = tf.split(output, 3, 1)
 
    ### softmax transform out_pi
    max_pi = tf.reduce_max(out_pi, 1, keepdims=True)
    out_pi = tf.subtract(out_pi, max_pi)
    out_pi = tf.exp(out_pi)
    normalize_pi = tfm.reciprocal(tf.reduce_sum(out_pi, 1, keepdims=True))
    out_pi = tf.multiply(normalize_pi, out_pi)
 
    ### For out_alpha and out_beta we just add an offset
    alpha_min = 2.0
    out_alpha = tfm.add(out_alpha,alpha_min)   
    
    beta_min = 0.1
    out_beta = tfm.add(out_beta,beta_min)
 
    return out_pi, out_alpha, out_beta
 
# function to get coefficients as numpy arrays
def get_mixture_coef_np(output):
    out_pi, out_alpha, out_beta=get_mixture_coef(output)
    return out_pi.numpy(), out_alpha.numpy(), out_beta.numpy()

 
# define network architecture with Keras layers
def Gamma_MDN(K,n_neurons,losses, input_shape, RI=True):
  """
  Helper function to define and compile the MDN model.
  We allow the posibility of using the Reliability Index (RI) as metric
  """
  # Mixture parameters
  KMIX = K  # number of mixtures
  NOUT = KMIX * 3  # KMIX times a pi, alpha and beta
 
  # number of neurons of each layer
  n_hidden_1 = n_neurons  # 1st layer
  n_hidden_2 = n_hidden_1/2  # 2nd layer
  n_hidden_3 = n_hidden_2/2  # 3rd layer

  # set initializer properties (to fix random seed)
  initializer = tf.keras.initializers.HeUniform(seed=2718)

  # initialize network and add layers
  model = Sequential()
  # add hidden layer 1
  model.add(Dense(n_hidden_1, input_shape=(input_shape,),
                  kernel_initializer=initializer, 
                  use_bias=False, name='Hidden_1'))
  model.add(BatchNormalization(name='Batch_1'))
  model.add(Activation('relu', name='ReLU_1'))
  # add hidden layer 2
  model.add(Dense(n_hidden_2,
                  kernel_initializer=initializer, 
                  use_bias=False, name='Hidden_2'))
  model.add(BatchNormalization(name='Batch_2'))
  model.add(Activation('relu', name='ReLU_2'))
  # add hidden layer 3
  model.add(Dense(n_hidden_3,
                  kernel_initializer=initializer, 
                  use_bias=False, name='Hidden_3'))
  model.add(BatchNormalization(name='Batch_3'))
  model.add(Activation('relu', name='ReLU_3'))
  # add output layer (the one for the mixture parameters)
  model.add(Dense(NOUT,
                  kernel_initializer=initializer,
                  name='Output'))
  model.add(Activation('softplus', name='Softplus'))
  # compile the model
  if RI:
    model.compile(loss=losses, metrics= RI_metric, optimizer='adam')
  else:
    model.compile(loss=losses, optimizer='adam')
  return model

# summarize the model
X_train=np.ones((100,10))
model = Gamma_MDN(4,128,'', X_train.shape[1],RI=False)
model.summary()

 
downloaded.GetContentFile('Y3_TRAIN_APRIL2018_NVP_Y4_new.fits')  
dat = Table.read('Y3_TRAIN_APRIL2018_NVP_Y4_new.fits', format='fits')
train_data_set = dat.to_pandas()

downloaded.GetContentFile('validsample_may2018_2_2_new.fits')  
dat = Table.read('validsample_may2018_2_2_new.fits', format='fits')
test_data_set = dat.to_pandas()

# Datasets are now stored in a Pandas Dataframe


train_data_set.head()


def gm_ll_loss(y_actual,y_pred):
    # get mixture coefficients    
    out_pi, out_alpha, out_beta = get_mixture_coef(y_pred)
    
    # define mixture distribution
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=out_pi),
        components_distribution=tfd.Gamma(
            concentration=out_alpha,       
            rate=out_beta))
    
    # Evaluate log-probability of y
    log_likelihood = gm.log_prob(tf.transpose(y_actual))     
    return -tf.reduce_mean(log_likelihood, axis=-1)

def PIT(y_actual,y_predict):
  # get mixture coefficients  
  out_pi, out_alpha, out_beta = get_mixture_coef(y_predict)   
  # define mixture distribution
  dist = tfd.Gamma(concentration=out_alpha, rate=out_beta)
  # calculate CDF at true redshift
  cdf = dist.cdf(y_actual)
  # take weights pi into account
  M_cdf = tfm.multiply(out_pi,cdf)
  # Return PIT=sum(pi*CDF)
  return tfm.reduce_sum(M_cdf, 1, keepdims=True)

def RI_metric(y_actual,y_predict):
  # calculate PIT values
  F_t=PIT(y_actual,y_predict)
  # get relative number of observations in each bin
  hist = tf.histogram_fixed_width(F_t, [0.0, 1.0], nbins=20, dtype=tf.dtypes.int32, name=None)
  k_t = tfm.divide(hist,tfm.reduce_sum(hist))
  # calculate relative difference from uniformity
  RI = tfm.abs(tfm.subtract(k_t,1/20))
  return tfm.reduce_sum(RI)

feat_sel = ['COADD_OBJECT_ID','Z','SOF_CM_T','SOF_CM_MAG_CORRECTED_G', 'SOF_CM_MAG_CORRECTED_R', 'SOF_CM_MAG_CORRECTED_I', 'SOF_CM_MAG_CORRECTED_Z', 'MAG_AUTO_CORRECTED_Y', 'ERR_Z', 'SOF_CM_T_ERR','SOF_CM_MAG_ERR_G', 'SOF_CM_MAG_ERR_R', 'SOF_CM_MAG_ERR_I', 'SOF_CM_MAG_ERR_Z', 'MAGERR_AUTO_Y']
df_train = train_data_set[feat_sel]

### Remove galaxies with magnitude errors < 0 and small redshifts (stars)
df_train = df_train[(df_train.Z<2.0) &
                    (df_train.SOF_CM_MAG_ERR_G>0.) &
                    (df_train.SOF_CM_MAG_ERR_R>0.) &
                    (df_train.SOF_CM_MAG_ERR_I>0.) &
                    (df_train.SOF_CM_MAG_ERR_Z>0.) &
                    (df_train.MAG_AUTO_CORRECTED_Y>0.) &
                    (df_train.MAG_AUTO_CORRECTED_Y<40.) &
                    (df_train.SOF_CM_T>0.)]


# let's change the column names for convenience
feat_names = {'COADD_OBJECT_ID': 'id','Z': 'redshift','SOF_CM_T': 'T','SOF_CM_MAG_CORRECTED_G': 'g', 'SOF_CM_MAG_CORRECTED_R': 'r', 'SOF_CM_MAG_CORRECTED_I': 'i', 'SOF_CM_MAG_CORRECTED_Z': 'z', 'MAG_AUTO_CORRECTED_Y': 'Y', 'ERR_Z': 'redshift_err', 'SOF_CM_MAG_ERR_G': 'g_err', 'SOF_CM_MAG_ERR_R': 'r_err', 'SOF_CM_MAG_ERR_I': 'i_err', 'SOF_CM_MAG_ERR_Z': 'z_err', 'MAGERR_AUTO_Y': 'Y_err'}
df_train=df_train.rename(columns=feat_names)


train_data_set.shape

df_train_aug = df_train[['id','redshift','T','g','r','i','z','Y']].copy()
col_names = ['g','r','i','z','Y']
for i in range(len(col_names)-1):
  for j in range(i+1,len(col_names)):
      df_train_aug[col_names[i]+'-'+col_names[j]]= df_train_aug[col_names[i]]-df_train_aug[col_names[j]]

# take log10 of magnitude errors
df_train_aug['T']=np.log10(df_train_aug['T'])

df_train_aug.head()


# function to plot variables
def plot_variables(data,feat_disp):
  fig, axes = plt.subplots(nrows=1, ncols=len(feat_disp), figsize=(20,3))
  for i in range(int(len(feat_disp))):
    # Density Plot and Histogram of all arrival delays
    sns.histplot(data[feat_disp[i]], 
                 element="step",
                 stat="density",
                 bins=int(180/5), 
                 color = 'coral',
                 fill = False, 
                 ax=axes[i])
    axes[i].set_xlabel(feat_disp[i])
    axes[0].set_ylabel('Density')
    #axes[i].set_yscale('log')
  plt.tight_layout()
  plt.show()
  


#features to plot. Magnitudes
feat_disp =['redshift','T','g','r','i','z','Y']
plot_variables(df_train_aug,feat_disp)

#features to plot. Color differences
feat_disp =['g-r','g-i','g-z','g-Y','r-i']
plot_variables(df_train_aug,feat_disp)

#features to plot. Color differences
feat_disp =['r-z','r-Y','i-z','i-Y','z-Y']
plot_variables(df_train_aug,feat_disp)



# Load the Drive helper and mount
 
# This will prompt for authorization.
drive.mount('/content/drive')
 
# Change path to file directory
os.chdir('/content/drive/My Drive/Colab Notebooks/gamma MDN photoz DES (Aurelio)')

sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1})


### The training features ###
feat_disp = ['redshift','T','g','r','i','z','Y','g-r','g-i','g-z','g-Y','r-i','r-z','r-Y','i-z','i-Y','z-Y']

X = df_train_aug.to_numpy()

df_X = pd.DataFrame(X[0:3000,1:],columns=feat_disp)


pair_colors_scaled = sns.pairplot(df_X, diag_kind="kde", height=1, aspect=1.5, corner=True, 
                                  plot_kws=dict(s=0.5, color = 'coral', edgecolor="coral"),
                                  diag_kws=dict(color='coral'))

#pair_colors_scaled.savefig("features_pair_plot_wT.pdf")

# dataframe with Y channel (w)
df_train_wY = df_train_aug.copy()

# dataframe without Y channel (woY)
df_train_woY = df_train_aug.drop(['Y','g-Y','r-Y','i-Y','z-Y'], axis=1)

### The training features with Y channel ###
feat_train = list(df_train_wY)

Xw = df_train_wY[feat_train[2:]].to_numpy()
yw = df_train_wY.redshift

### The training features with Y channel ###
feat_train = list(df_train_woY)

Xwo = df_train_woY[feat_train[2:]].to_numpy()
ywo = df_train_woY.redshift
Xwo.shape


# Reinitiate the transformer pipeline
steps_w = [('scaler', RobustScaler()), ('pca', PCA())]

steps_wo = [('scaler', RobustScaler()), ('pca', PCA())]

pipeline_w = Pipeline(steps_w)

pipeline_wo = Pipeline(steps_wo)

# fit and transform the whole dataset
X_pca_w=pipeline_w.fit_transform(Xw)
X_pca_wo=pipeline_wo.fit_transform(Xwo)

eigenvec = pipeline_w.named_steps.pca.components_
eigenval = pipeline_w.named_steps.pca.explained_variance_ratio_
sample = np.random.randint(0,X_pca_w.shape[0],6000)
x_max = np.max(X_pca_w[sample,0])
y_max = np.max(X_pca_w[sample,1])
z_max = np.max(X_pca_w[sample,2])
x_min = np.min(X_pca_w[sample,0])
y_min = np.min(X_pca_w[sample,1])
z_min = np.min(X_pca_w[sample,2])

fig, ax = plt.subplots(1,3,figsize=(18,6))
ax[0].scatter(X_pca_w[sample,0],X_pca_w[sample,1], color='coral', s=0.5)
for i in range(X_pca_w.shape[1]):
        ax[0].arrow(0, 0, eigenvec[0,i]*0.7*x_max,eigenvec[1,i]*0.7*y_max,color='tab:blue',alpha=0.5, width=0.01, head_width=0.1)
        ax[0].text(eigenvec[0,i]*x_max, eigenvec[1,i]*y_max,
            list(df_train_wY)[i+2], color='tab:blue')
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')


ax[1].scatter(X_pca_w[sample,0],X_pca_w[sample,2], color='coral', s=0.5)
for i in range(X_pca_w.shape[1]):
        ax[1].arrow(0, 0, eigenvec[0,i]*0.7*x_max,eigenvec[2,i]*0.7*z_max,color='tab:blue',alpha=0.5, width=0.01, head_width=0.1)
        ax[1].text(eigenvec[0,i]*x_max, eigenvec[2,i]*z_max,
            list(df_train_wY)[i+2], color='tab:blue')
ax[1].set_xlabel('PC1')
ax[1].set_ylabel('PC3')

ax[2].scatter(X_pca_w[sample,1],X_pca_w[sample,2], color='coral', s=0.5)
for i in range(X_pca_w.shape[1]):
        ax[2].arrow(0, 0, eigenvec[1,i]*0.7*y_max,eigenvec[2,i]*0.7*z_max,color='tab:blue',alpha=0.5, width=0.01, head_width=0.1)
        ax[2].text(eigenvec[1,i]*y_max, eigenvec[2,i]*z_max,
            list(df_train_wY)[i+2], color='tab:blue')
ax[2].set_xlabel('PC2')
ax[2].set_ylabel('PC3')
        
# set axis limits
lim_x=max(np.abs(x_min),x_max)
lim_y=max(np.abs(y_min),y_max)
lim_z=max(np.abs(z_min),z_max)

ax[0].set_xlim(-lim_x,lim_x)
ax[0].set_ylim(-lim_y,lim_y)

ax[1].set_xlim(-lim_x,lim_x)
ax[1].set_ylim(-lim_z,lim_z)

ax[2].set_xlim(-lim_y,lim_y)
ax[2].set_ylim(-lim_z,lim_z)

plt.tight_layout()
plt.show()

var_w = pipeline_w.named_steps.pca.explained_variance_ratio_*100
cum_var_w = pipeline_w.named_steps.pca.explained_variance_ratio_.cumsum()*100

var_wo = pipeline_wo.named_steps.pca.explained_variance_ratio_*100
cum_var_wo = pipeline_wo.named_steps.pca.explained_variance_ratio_.cumsum()*100

print("Percentage of variability explained by the PCs:")
print('            With Y Channel         Without Y Channel')
print("           Var       Cum_var       Var       Cum_var")
for i in range(X_pca_w.shape[1]):
  ind_pc=i+1
  if i <  X_pca_wo.shape[1]:
    print('PC %.0f:    %.5f     %.5f     %.5f     %.5f'   %(ind_pc,var_w[i],cum_var_w[i],var_wo[i],cum_var_wo[i]))
  else:
    print('PC %.0f:    %.5f     %.5f           -            -'   %(ind_pc,var_w[i],cum_var_w[i]))


d_w={"x": np.arange(1,X_pca_w.shape[1]+1),"Var": var_w,"Cum_var": cum_var_w}
df_pc_w = pd.DataFrame(data=d_w)

d_wo={"x": np.arange(1,X_pca_wo.shape[1]+1),"Var": var_wo,"Cum_var": cum_var_wo}
df_pc_wo = pd.DataFrame(data=d_wo)

ig, ax = plt.subplots(1,2,figsize=(15,5))

# With Y channel
ax2 = ax[0].twinx() #This allows the common axes (flow rate) to be shared
sns.lineplot(x="x", y="Var",
              marker="o", 
             color = "coral",
             data = df_pc_w, ax=ax[0])
sns.lineplot(x="x", y="Cum_var",
              marker="o", 
             color = 'black',
             data = df_pc_w, ax=ax2)
#ax.set(xlabel='Principal Component', ylabel='Explained Variance (%)')
ax2.set(xlabel='Principal Component', ylabel='Explained Cumulative \n Variance (%)')
ax[0].set_ylabel(ylabel='Explained Variance (%)', color='coral', fontsize=15)  # we already handled the x-label with ax1
ax[0].set_xlabel(xlabel='Principal Component', fontsize=15)  # we already handled the x-label with ax1
ax[0].set_title('With Y channel', style='italic', fontsize=14)
ax[0].tick_params(axis='y', labelcolor='coral', labelsize=13)
ax[0].xaxis.set_ticks(np.arange(0,X_pca_w.shape[1]+2,2))
#tick_labels = np.arange(0,22,2.5)
#ax.set_xticklabels(tick_labels.astype(int))
ax2.set_ylabel(ylabel='Accumulated Explained \n Variance (%)', color='black', fontsize=15)  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelsize=13)

#Without Y channel
ax4 = ax[1].twinx() #This allows the common axes (flow rate) to be shared
sns.lineplot(x="x", y="Var",
              marker="o", 
             color = "coral",
             data = df_pc_wo, ax=ax[1])
sns.lineplot(x="x", y="Cum_var",
              marker="o", 
             color = 'black',
             data = df_pc_wo, ax=ax4)
#ax.set(xlabel='Principal Component', ylabel='Explained Variance (%)')
ax4.set(xlabel='Principal Component', ylabel='Explained Cumulative \n Variance (%)')
ax[1].set_ylabel(ylabel='Explained Variance (%)', color='coral', fontsize=15)  # we already handled the x-label with ax1
ax[1].set_xlabel(xlabel='Principal Component', fontsize=15)  # we already handled the x-label with ax1
ax[1].set_title('Without Y channel', style='italic', fontsize=14)
ax[1].tick_params(axis='y', labelcolor='coral', labelsize=13)
#ax[1].xaxis.set_ticks(np.arange(0,X_pca_wo.shape[1]+2,2))
#tick_labels = np.arange(0,22,2.5)
#ax.set_xticklabels(tick_labels.astype(int))
ax4.set_ylabel(ylabel='Accumulated Explained \n Variance (%)', color='black', fontsize=15)  # we already handled the x-label with ax1
ax4.tick_params(axis='y', labelsize=13)

plt.tight_layout()
#plt.savefig("PC_variability_wT.pdf")
plt.show()

pair_colors_scaled = sns.pairplot(pd.DataFrame(X_pca_w[1:5000,0:4],columns=['PC1','PC2','PC3','PC4']), diag_kind="kde", height=1.5, aspect=1.5, corner=False, 
                                  plot_kws=dict(s=0.5, color = 'coral', edgecolor="coral"),
                                  diag_kws=dict(color='coral'))

#pair_colors_scaled.savefig("PCs_correlation_wT.pdf")

# remove column with redshift error (let Y changel)
#df_train_wY = df_train_aug.drop(['redshift_err'], axis=1)
df_train_wY = df_train_aug.copy()

# dataframe without Y channel (woY)
#df_train_woY = df_train_aug.drop(['redshift_err','Y','Y_err','g-Y','g-Y_err','r-Y','r-Y_err','i-Y','i-Y_err','z-Y','z-Y_err'], axis=1)
df_train_woY = df_train_aug.drop(['Y','g-Y','r-Y','i-Y','z-Y'], axis=1)

### The training features with Y channel ###
feat_train = list(df_train_wY)

Xw = df_train_wY[feat_train[2:]].to_numpy()
yw = df_train_wY.redshift

### The training features with Y channel ###
feat_train = list(df_train_woY)

Xwo = df_train_woY[feat_train[2:]].to_numpy()
ywo = df_train_woY.redshift

Xw.shape

# Reinitiate the transformer pipeline
steps_w = [('scaler', RobustScaler()), ('pca', PCA())]

steps_wo = [('scaler', RobustScaler()), ('pca', PCA())]

pipeline_w = Pipeline(steps_w)

pipeline_wo = Pipeline(steps_wo)

# parameters to initiate and train the gamma-MDN

# number of mixtures
K= 5
# number of neurons for first hidden layer
n_hidden_1=128
# batch_size
batch_size = 256*2
# number of epochs
n_epoch=900 # 400 is more than enough
# number of K folds
n_folds=3
# validation frequency
val_freq=1


# define training and test partitions
X_training_w, Xtest_w, Y_training_w, Ytest_w = train_test_split(Xw, yw, test_size=15/100, shuffle=True, random_state=1)

# define training and test partitions
X_training_wo, Xtest_wo, Y_training_wo, Ytest_wo = train_test_split(Xwo, ywo, test_size=15/100, shuffle=True, random_state=1)

checkpoint_path = "/content/drive/MyDrive/test_in_original_withY_model_callback.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)


# Define the K-fold Cross Validator
kfold_w = KFold(n_splits=n_folds, shuffle=True, random_state=0)

# K-fold Cross Validation model evaluation
fold_no = 1
isFirst = True
for train, val in kfold_w.split(X_training_w, Y_training_w):
 
   # fit and transform the training partition
  X_train=pipeline_w.fit_transform(X_training_w[train])
  Y_train=Y_training_w[train, np.newaxis]
  # transform validation partition
  X_val=pipeline_w.transform(X_training_w[val])
  Y_val=Y_training_w[val, np.newaxis]

  # define and compile model  
  model_w_cv = Gamma_MDN(K,n_hidden_1,gm_ll_loss, X_train.shape[1], True) 

  # Generate a print
  print('------------------------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} out of {n_folds}')
  # Increase fold number
  fold_no = fold_no + 1

  # fit model to data
  with tf.device('/device:GPU:0'):
    history_w_cv=model_w_cv.fit(X_train, Y_train, 
                      validation_data=(X_val,Y_val),# for first validation partition
                      epochs=n_epoch,
                      batch_size = batch_size,
                      verbose=1,
                      validation_freq=val_freq,
                      callbacks=[cp_callback])

  # evaluate and store performance
  if isFirst:
    isFirst = False
    cv_train_loss_w_cv = history_w_cv.history['loss']
    cv_val_loss_w_cv = history_w_cv.history['val_loss']
    cv_train_RI_w_cv = history_w_cv.history['RI_metric']
    cv_val_RI_w_cv = history_w_cv.history['val_RI_metric']
  else: 

    cv_train_loss_w_cv = np.vstack((cv_train_loss_w_cv,history_w_cv.history['loss']))
    cv_val_loss_w_cv = np.vstack((cv_val_loss_w_cv,history_w_cv.history['val_loss']))
    cv_train_RI_w_cv = np.vstack((cv_train_RI_w_cv,history_w_cv.history['RI_metric']))
    cv_val_RI_w_cv = np.vstack((cv_val_RI_w_cv,history_w_cv.history['val_RI_metric']))
    

checkpoint_path = "/content/drive/MyDrive/test_in_original_noY_model_callback.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)

# Define the K-fold Cross Validator
kfold_wo = KFold(n_splits=n_folds, shuffle=True, random_state=0)

# K-fold Cross Validation model evaluation
fold_no = 1
isFirst = True
for train, val in kfold_wo.split(X_training_wo, Y_training_wo):
 
   # fit and transform the training partition
  X_train=pipeline_wo.fit_transform(X_training_wo[train])
  Y_train=Y_training_wo[train, np.newaxis]
  # transform validation partition
  X_val=pipeline_wo.transform(X_training_wo[val])
  Y_val=Y_training_wo[val, np.newaxis]

  # define and compile model  
  model_wo_cv = Gamma_MDN(K,n_hidden_1,gm_ll_loss, X_train.shape[1], True) 

  # Generate a print
  print('------------------------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} out of {n_folds}')
  # Increase fold number
  fold_no = fold_no + 1

  # fit model to data
  with tf.device('/device:GPU:0'):
    history_wo_cv=model_wo_cv.fit(X_train, Y_train, 
                      validation_data=(X_val,Y_val),# for first validation partition
                      epochs=n_epoch,
                      batch_size = batch_size,
                      verbose=1,
                      validation_freq=val_freq,
                      callbacks=[cp_callback])

  # evaluate and store performance
  if isFirst:
    isFirst = False
    cv_train_loss_wo_cv = history_wo_cv.history['loss']
    cv_val_loss_wo_cv = history_wo_cv.history['val_loss']
    cv_train_RI_wo_cv = history_wo_cv.history['RI_metric']
    cv_val_RI_wo_cv = history_wo_cv.history['val_RI_metric']
  else: 

    cv_train_loss_wo_cv = np.vstack((cv_train_loss_wo_cv,history_wo_cv.history['loss']))
    cv_val_loss_wo_cv = np.vstack((cv_val_loss_wo_cv,history_wo_cv.history['val_loss']))
    cv_train_RI_wo_cv = np.vstack((cv_train_RI_wo_cv,history_wo_cv.history['RI_metric']))
    cv_val_RI_wo_cv = np.vstack((cv_val_RI_wo_cv,history_wo_cv.history['val_RI_metric']))
    


# epochs arrays
epochs_train = history_wo_cv.epoch+np.ones(len(history_wo_cv.epoch))
epochs_val = history_wo_cv.epoch+val_freq*np.ones(len(history_wo_cv.epoch))

## to avoid nans, we use nanmean and nanstd
# losses arrays with Y channel
#mean_train_loss_w_cv=np.nanmean(cv_train_loss_w_cv[1:,:],axis=0)
#mean_val_loss_w_cv=np.nanmean(cv_val_loss_w_cv[1:,:],axis=0)
#mean_train_RI_w_cv=np.nanmean(cv_train_RI_w_cv[1:,:],axis=0)
#mean_val_RI_w_cv=np.nanmean(cv_val_RI_w_cv[1:,:],axis=0)


# losses arrays for log-likelihood. K=5
mean_train_loss_wo_cv=np.nanmean(cv_train_loss_wo_cv[1:,:],axis=0)
mean_val_loss_wo_cv=np.nanmean(cv_val_loss_wo_cv[1:,:],axis=0)
mean_train_RI_wo_cv=np.nanmean(cv_train_RI_wo_cv[1:,:],axis=0)
mean_val_RI_wo_cv=np.nanmean(cv_val_RI_wo_cv[1:,:],axis=0)

# optimal epoch
#epoch_min_w_cv=np.argmin(mean_val_loss_w_cv)+1
#epoch_min_RI_w_cv=np.argmin(mean_val_RI_w_cv)+1
epoch_min_wo_cv=np.argmin(mean_val_loss_wo_cv)+1
epoch_min_wo_cv_wo=np.argmin(mean_val_RI_wo_cv)+1

# Plot training & validation loss values
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 3.5))

# plot for crps
# losses
color_1 = "coral"
color_2 = "tab:blue"
#ax[0].plot(epochs_train,mean_train_loss_w_cv,c=color_1)
#ax[0].plot(epochs_val,mean_val_loss_w_cv,c=color_2)
#ax[0].axvline(x=epoch_min_w_cv, color='black', alpha=1)
#ax[0].set_ylim(np.min(mean_train_loss_w_cv),mean_val_loss_w_cv[0])
#ax[0].set_title('$\gamma$-MDN with Y channel', style='italic', fontsize=14)
#ax[0].set_ylabel('Loss', fontsize=15)
#ax[0].set_xlabel('Epoch', fontsize=15)
#ax[0].legend(['Training', 'Validation'], loc='upper right', fontsize=13)
#ax[0].grid()

# RI
#ax[1].plot(epochs_train,mean_train_RI_w_cv,c=color_1)
#ax[1].plot(epochs_val,mean_val_RI_w_cv,c=color_2)
#ax[1].axvline(x=epoch_min_RI_w_cv, color='black', alpha=1)
#ax[1].set_ylim(np.min(mean_val_RI_w_cv),mean_val_RI_w_cv[0])
#ax[1].set_yscale('log')
#ax[1].set_title('$\gamma$-MDN with Y channel', style='italic', fontsize=14)
#ax[1].set_ylabel('RI', style='italic', fontsize=15)
#ax[1].set_xlabel('Epoch', fontsize=15)
#ax[1].legend(['Training', 'Validation'], loc='upper right', fontsize=13)
#ax[1].grid()

# plot for LL
ax[2].plot(epochs_train,mean_train_loss_wo_cv,c=color_1)
ax[2].plot(epochs_val,mean_val_loss_wo_cv,c=color_2)
ax[2].axvline(x=epoch_min_wo_cv, color='black', alpha=1)

ax[2].set_title('$\gamma$-MDN without Y channel', style='italic', fontsize=14)
ax[2].set_ylabel('Loss', fontsize=15)
ax[2].set_xlabel('Epoch', fontsize=15)
ax[2].legend(['Training', 'Validation'], loc='upper right', fontsize=13)
ax[2].grid()

# RI
ax[3].plot(epochs_train,mean_train_RI_wo_cv,c=color_1)
ax[3].plot(epochs_val,mean_val_RI_wo_cv,c=color_2)
ax[3].axvline(x=epoch_min_wo_cv, color='black', alpha=1)

ax[3].set_title('$\gamma$-MDN without Y channel', style='italic', fontsize=14)
ax[3].set_ylabel('RI', style='italic', fontsize=15)
ax[3].set_xlabel('Epoch', fontsize=15)
ax[3].legend(['Training', 'Validation'], loc='upper right', fontsize=13)
ax[3].grid()

hight = 0.90
#fig.text(0.01, hight, "a)", horizontalalignment='left', verticalalignment='center', fontsize=15)
#fig.text(0.265, hight, "b)", horizontalalignment='left', verticalalignment='center', fontsize=15)
#fig.text(0.505, hight, "c)", horizontalalignment='left', verticalalignment='center', fontsize=15)
#fig.text(0.76, hight, "d)", horizontalalignment='left', verticalalignment='center', fontsize=15)


plt.tight_layout()
#plt.savefig("LL_RI_vs_epochs_wT.pdf")
plt.show()

#print("For model with Y channel: %0.f" %(np.argmin(mean_val_loss_w_cv)+1))
print("")
print("For model without Y channel: %0.f" %(np.argmin(mean_val_loss_wo_cv)+1))

###### With Y channel
# Create a callback that saves the model's weights
checkpoint_path = "/content/drive/MyDrive/PDFphotoz/aure_original_second_callback_withY.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)

# fit and transform the training partition
X_train_w=pipeline_w.fit_transform(X_training_w)

# transform validation partition
X_test_w=pipeline_w.transform(Xtest_w)

Y_train_w = Y_training_w[:, np.newaxis]  # add extra axis as tensorflow expects this 
Y_test_w = Ytest_w[:, np.newaxis]

# Initialize and compile model (do not comput RI, it crashes)
model_w = Gamma_MDN(K, n_hidden_1, gm_ll_loss, X_train_w.shape[1], True)

# fit the model with the optimal number of epochs
#n_epoch_w=np.argmin(mean_val_loss_w)+1

n_epoch_w = 850

with tf.device('/device:GPU:0'):
    history_w=model_w.fit(X_train_w, Y_train_w, 
                      epochs=n_epoch_w,
                      batch_size = batch_size,
                      verbose=1,
                      callbacks=[cp_callback])



###### Without Y channel
checkpoint_path = "/content/drive/MyDrive/PDFphotoz/aure_original_second_callback_noY.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)

# fit and transform the training partition
X_train_wo=pipeline_wo.fit_transform(X_training_wo)

# transform validation partition
X_test_wo=pipeline_wo.transform(Xtest_wo)

Y_train_wo = Y_training_wo[:, np.newaxis]  # add extra axis as tensorflow expects this 
Y_test_wo = Ytest_wo[:, np.newaxis]

# Initialize and compile model (do not comput RI, it crashes)
model_wo = Gamma_MDN(K, n_hidden_1, gm_ll_loss, X_train_wo.shape[1], True)

# fit the model with the optimal number of epochs
#n_epoch_wo=np.argmin(mean_val_loss_wo)+1

n_epoch_wo = 850

with tf.device('/device:GPU:0'):
    history_wo=model_wo.fit(X_train_wo, Y_train_wo, 
                      epochs=n_epoch_wo,
                      batch_size = batch_size,
                      verbose=1,
                      callbacks=[cp_callback])



### With Y channel
# compute PIT and RI
# for the train
output=model_w.predict(X_train_w)  
F_t_train_w=PIT(Y_train_w,output)
RI_train_w=RI_metric(Y_train_w,output).numpy()

# for the test
output=model_w.predict(X_test_w) 
F_t_test_w=PIT(Y_test_w,output)
RI_test_w=RI_metric(Y_test_w,output).numpy()

### Without Y channel
# compute PIT and RI
# for the train
output=model_wo.predict(X_train_wo)  
F_t_train_wo=PIT(Y_train_wo,output)
RI_train_wo=RI_metric(Y_train_wo,output).numpy()

# for the test
output=model_wo.predict(X_test_wo) 
F_t_test_wo=PIT(Y_test_wo,output)
RI_test_wo=RI_metric(Y_test_wo,output).numpy()


fig = plt.figure(figsize=(15, 3.5))
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)
ax4 = fig.add_subplot(144)

n_bins_train=20
sns.histplot(F_t_train_w.numpy(),
             bins=n_bins_train,
             stat="density",
             color = 'coral', 
             legend = False,
             ax=ax1)

n_bins_test=20
sns.histplot(F_t_test_w.numpy(), 
             bins=n_bins_test,
             stat="density",
             color = 'tab:blue',  
             legend = False,
             ax=ax2)

sns.histplot(F_t_train_wo.numpy(), 
             bins=n_bins_train, 
             stat="density",
             color = 'orange', 
             legend = False,
             ax=ax3)

n_bins_test=20
sns.histplot(F_t_test_wo.numpy(),  
             bins=n_bins_test, 
             stat="density",
             color = 'tab:blue', 
             legend = False,
             ax=ax4)

ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax3.set_ylim(bottom=0)
ax4.set_ylim(bottom=0)

ax1.set_xlabel('PIT', fontsize=15)
ax1.set_title(' Train ($RI$ = %.4f)' %(RI_train_w), style='italic', fontsize=14)
ax1.set_xlabel('PIT', fontsize=15)
ax2.set_title(' Test ($RI$ = %.4f)' %(RI_test_w), style='italic', fontsize=14)
ax1.set_ylabel('Relative Frequency', fontsize=15)
ax2.set_xlabel('PIT', fontsize=15)
ax2.set_ylabel('Relative Frequency', fontsize=15)

ax3.set_xlabel('PIT', fontsize=15)
ax3.set_title(' Train ($RI$ = %.4f)' %(RI_train_wo), style='italic', fontsize=14)
ax3.set_xlabel('PIT', fontsize=15)
ax4.set_title(' Test ($RI$ = %.4f)' %(RI_test_wo), style='italic', fontsize=14)
ax3.set_ylabel('Relative Frequency', fontsize=14)
ax4.set_xlabel('PIT', fontsize=15)
ax4.set_ylabel('Relative Frequency', fontsize=14)

ax1.tick_params(axis='y', labelsize=13)
ax1.tick_params(axis='x', labelsize=13)
ax2.tick_params(axis='y', labelsize=13)
ax2.tick_params(axis='x', labelsize=13)
ax3.tick_params(axis='y', labelsize=13)
ax3.tick_params(axis='x', labelsize=13)
ax4.tick_params(axis='y', labelsize=13)
ax4.tick_params(axis='x', labelsize=13)

height = 0.97
fig.text(0.01, height, "a)", horizontalalignment='left', verticalalignment='center', fontsize=15)
fig.text(0.505, height, "b)", horizontalalignment='left', verticalalignment='center', fontsize=15)

fig.text(0.22, height, "$\gamma$-MDN with Y Channel", horizontalalignment='left', verticalalignment='center', fontsize=14, style='italic')
fig.text(0.72, height, "$\gamma$-MDN without Y Channel", horizontalalignment='left', verticalalignment='center', fontsize=14, style='italic')

plt.tight_layout()
#plt.savefig("PIT_train_wT.pdf")
plt.show()



## with Y channel
# for the train
output=model_w.predict(X_train_w)
pi_tr_w, alpha_tr_w, beta_tr_w = get_mixture_coef_np(output)

# for the test
output=model_w.predict(X_test_w)
pi_ts_w, alpha_ts_w, beta_ts_w = get_mixture_coef_np(output)

## Without Y channel
# for the train
output=model_wo.predict(X_train_wo)
pi_tr_wo, alpha_tr_wo, beta_tr_wo = get_mixture_coef_np(output)

# for the test
output=model_wo.predict(X_test_wo)
pi_ts_wo, alpha_ts_wo, beta_ts_wo = get_mixture_coef_np(output)


def pdf_mode(pis, alphas, betas):
    """ Helper function to find the mode of the mixture PDF"""
    
    # define mixture distribution
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=pis),
        components_distribution=tfd.Gamma(
            concentration=alphas,       
            rate=betas))

    # compute modes of each component
    modes = (alphas - 1.0) / betas
       
    # find value of mixture PDF at components modes
    isFirst=True
    for i in range(len(pis[0])):
      if isFirst:
        gm_modes = gm.prob(modes[:,0])
        isFirst=False
      else:
        gm_modes = np.vstack([gm_modes,gm.prob(modes[:,i])])
    
    # find argument of mode giving largest PDF
    mode_arg = np.argmax(gm_modes,axis=0)

    # find corresponding photo_z
    mode = np.ones(pis.shape[0])
    for i in range(pis.shape[0]):
      mode[i] = modes[i,mode_arg[i]]

    return mode

def return_df(pis, alphas, betas, Y_valid):
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
    temp = np.concatenate((pis, alphas, betas, alphas/betas, np.sqrt(alphas)/betas), axis=1)
    df = pd.DataFrame(temp, columns=names)
        
    variances = alphas/betas**2
    means = (alphas / betas) 
    modes = pdf_mode(pis,alphas,betas)
        
    df['Mode'] = modes
    df['Mean'] = np.average(means, weights=pis, axis=1)
    df['variance'] =  np.average(means**2 + variances**2, weights=pis, axis=1) - df['Mean'].values**2
    df['stddev'] = np.sqrt(df.variance)
    df['CoV'] = df['stddev']/df['Mean']
    df['redshift'] = Y_valid
    return df


## CRPS
# for the train
res_tr_w = return_df(pi_tr_w, alpha_tr_w, beta_tr_w, Y_train_w)
# for the test
res_ts_w = return_df(pi_ts_w, alpha_ts_w, beta_ts_w, Y_test_w)

## Log-likelihood
# for the train
res_tr_wo = return_df(pi_tr_wo, alpha_tr_wo, beta_tr_wo, Y_train_wo)
# for the test
res_ts_wo = return_df(pi_ts_wo, alpha_ts_wo, beta_ts_wo, Y_test_wo)

res_ts_wo.head(5)

fig, ax = plt.subplots(1,2, figsize=(9,5))
ax[0].hist(pi_ts_w.flatten(), density=True)
ax[1].hist(pi_ts_wo.flatten(), density=True)
ax[0].set_ylabel('Frequency')
ax[0].set_xlabel('Mixture weights \pi')
ax[1].set_xlabel('Mixture weights \pi')
ax[0].set_title('With Y channel')
ax[1].set_title('Without Y channel')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1,4, figsize=(18,5))
ax[0].hist(beta_ts_w.flatten(), density=True)
ax[1].hist(beta_ts_wo.flatten(), density=True)
ax[0].set_ylabel('Frequency')
ax[0].set_xlabel('Rate beta')
ax[1].set_xlabel('Rate beta')
ax[0].set_title('With Y Channel')
ax[1].set_title('Without Y Channel')

ax[2].hist(alpha_ts_w.flatten(), density=True)
ax[3].hist(alpha_ts_wo.flatten(), density=True)
ax[2].set_ylabel('Frequency')
ax[2].set_xlabel('Concentration alpha')
ax[3].set_xlabel('Concentration alpha')
ax[2].set_title('With Y Channel')
ax[3].set_title('Without Y Channel')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1,2, figsize=(10,5))
CoV_w=res_ts_w['stddev']/res_ts_w['Mean']*100
CoV_wo=res_ts_wo['stddev']/res_ts_wo['Mean']*100
ax[0].hist(CoV_w[CoV_w<20], density=True)
ax[1].hist(CoV_wo[CoV_wo<20], density=True)
ax[0].set_ylabel('Frequency')
ax[0].set_xlabel('Coefficient of Variation (%)')
ax[1].set_xlabel('Coefficient of Variation (%)')
ax[0].set_title('With Y Channel')
ax[1].set_title('Without Y Channel')

plt.tight_layout()

CoV_w_trun=CoV_w[CoV_w<20].copy()
CoV_wo_trun=CoV_wo[CoV_wo<20].copy()

CoV_wo.shape

CoV_wo_trun.shape

def plot_gamma_mix(pis, alphas, betas, ax, color='red', comp=True):
    """
    Plots the mixture of gamma models to axis=ax
    """
    x = np.linspace(0.0, 5, 600)
    final = np.zeros_like(x)
    for i, (weight_mix, alpha_mix, beta_mix) in enumerate(zip(pis, alphas, betas)):
        dist = tfd.Gamma(concentration=alpha_mix, rate=beta_mix)
        pdf = dist.prob(x)
        temp = pdf.numpy() * weight_mix
        final = final + temp
    ax.plot(x, final, c=color)

num_cols=6
num_rows=3
num_gal=num_cols*num_rows
gal_id = np.random.randint(1,len(Y_test_wo),num_gal)


#pi_ts_wo, alpha_ts_wo, beta_ts_wo

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 10))
count = 0
for i in range(num_rows):
  for j in range(num_cols):
    plot_gamma_mix(pi_ts_wo[gal_id][count], alpha_ts_wo[gal_id][count], beta_ts_wo[gal_id][count], axes[i,j], 'coral') 
    axes[i, j].axvline(x=Y_test_wo[gal_id[count]], color='black', alpha=0.7)
    axes[i, j].set_xlim(0,1.5)
    axes[i, j].set_ylabel('PDF', fontsize=14)
    axes[i, j].set_title('Test source ' + str(gal_id[count]), fontsize=14)
    axes[i, j].set_xlabel('Redshift', fontsize=14)
    count += 1    

axes[0, 0].legend(['log-like', 'Truth'], loc ='best', fontsize=12)
fig.tight_layout()

num_cols=8
num_rows=4
num_gal=int(num_cols*num_rows/2)

## In terms of CRPS
variance_wo=res_ts_wo['variance'].values
i_sort_wo=np.argsort(variance_wo)


# galaxies IDs
gal_id_wo = i_sort_wo[-num_gal:]

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 10))
count_wo = 0
count_ll = 0

for i in range(num_rows):
  for j in range(num_cols):
    if i<num_rows/2:
      plot_gamma_mix(pi_ts_wo[gal_id_wo][count_wo], alpha_ts_wo[gal_id_wo][count_wo],
                     beta_ts_wo[gal_id_wo][count_wo], axes[i,j], 'coral')  
      axes[i, j].axvline(x=Y_test_wo[gal_id_wo[count_wo]], color='black', alpha=0.7)
      axes[i, j].set_ylabel('PDF', fontsize=14)
      axes[i, j].set_title('Test source ' + str(gal_id_wo[count_wo]), fontsize=14)
      axes[i, j].set_xlabel('Redshift', fontsize=14)
      axes[i, j].set_xlim(0,1.5)

      count_wo += 1
    else:
      plot_gamma_mix(pi_ts_wo[gal_id_wo][count_ll], alpha_ts_wo[gal_id_wo][count_ll], 
                     beta_ts_wo[gal_id_wo][count_ll], axes[i,j], 'blue')  
      axes[i, j].axvline(x=Y_test_wo[gal_id_wo[count_ll]], color='black', alpha=0.7)
      axes[i, j].set_ylabel('PDF', fontsize=14)
      axes[i, j].set_title('Test source ' + str(gal_id_wo[count_ll]), fontsize=14)
      axes[i, j].set_xlabel('Redshift', fontsize=14)
      axes[i, j].set_xlim(0,1.5)

      count_ll += 1

axes[0, 0].legend(['log-like', 'Truth'], loc ='upper right', fontsize=12)
axes[2, 0].legend(['log-like', 'Truth'], loc ='upper right', fontsize=12)
fig.tight_layout()


# plot all test predictions and 5 times as much train predictions
Y_test=Y_test_w
Y_train=Y_train_w

amount = len(Y_test) 
p_ind_train = np.random.randint(0,len(Y_train),5*amount)
p_ind_test = np.random.randint(0,len(Y_test),amount)


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))
## With Y channel
# mean
axes[0,0].scatter(Y_train[p_ind_train], res_tr_w['Mean'].iloc[p_ind_train],
                alpha=0.5,s=4, c='coral')
axes[0,1].scatter(Y_train[p_ind_train], res_tr_w['Mean'].iloc[p_ind_train],
                alpha=0.5,s=4, c='coral')
axes[0,2].scatter(Y_test[p_ind_test], res_ts_w['Mean'].iloc[p_ind_test],
                alpha=0.5,s=4, c='coral')
axes[0,3].scatter(Y_test[p_ind_test], res_ts_w['Mean'].iloc[p_ind_test],
                alpha=0.5,s=4, c='coral')


## With Y channel
# mean
axes[1,0].scatter(Y_train[p_ind_train], res_tr_wo['Mean'].iloc[p_ind_train],
                alpha=0.5,s=4, c='coral')
axes[1,1].scatter(Y_train[p_ind_train], res_tr_wo['Mean'].iloc[p_ind_train],
                alpha=0.5,s=4, c='coral')
axes[1,2].scatter(Y_test[p_ind_test], res_ts_wo['Mean'].iloc[p_ind_test],
                alpha=0.5,s=4, c='coral')
axes[1,3].scatter(Y_test[p_ind_test], res_ts_wo['Mean'].iloc[p_ind_test],
                alpha=0.5,s=4, c='coral')


# add bisector lines
axes[0,0].plot([0, max(Y_train)], [0, max(Y_train)], linewidth=1, color='k')
axes[0,1].plot([0, max(Y_test)], [0, max(Y_test)], linewidth=1, color='k')
axes[0,2].plot([0, max(Y_train)], [0, max(Y_train)], linewidth=1, color='k')
axes[0,3].plot([0, max(Y_test)], [0, max(Y_test)], linewidth=1, color='k')

axes[1,0].plot([0, max(Y_train)], [0, max(Y_train)], linewidth=1, color='k')
axes[1,1].plot([0, max(Y_test)], [0, max(Y_test)], linewidth=1, color='k')
axes[1,2].plot([0, max(Y_train)], [0, max(Y_train)], linewidth=1, color='k')
axes[1,3].plot([0, max(Y_test)], [0, max(Y_test)], linewidth=1, color='k')

# set axis limints
axes[0,0].set_xlim(0, max(Y_train))
axes[0,0].set_ylim(0, max(Y_train))
axes[0,1].set_xlim(min(Y_train), max(Y_train))
axes[0,1].set_ylim(min(Y_train), max(Y_train))
axes[0,2].set_xlim(0, max(Y_test))
axes[0,2].set_ylim(0, max(Y_test))
axes[0,3].set_xlim(min(Y_test), max(Y_test))
axes[0,3].set_ylim(min(Y_test), max(Y_test))
axes[0,1].set_xscale('log')
axes[0,1].set_yscale('log')
axes[0,3].set_xscale('log')
axes[0,3].set_yscale('log')

axes[1,0].set_xlim(0, max(Y_train))
axes[1,0].set_ylim(0, max(Y_train))
axes[1,1].set_xlim(min(Y_train), max(Y_train))
axes[1,1].set_ylim(min(Y_train), max(Y_train))
axes[1,2].set_xlim(0, max(Y_test))
axes[1,2].set_ylim(0, max(Y_test))
axes[1,3].set_xlim(min(Y_test), max(Y_test))
axes[1,3].set_ylim(min(Y_test), max(Y_test))
axes[1,1].set_xscale('log')
axes[1,1].set_yscale('log')
axes[1,3].set_xscale('log')
axes[1,3].set_yscale('log')

# add title and axis labels
axes[0,0].set_title('With Y Train', fontsize=18)
axes[0,1].set_title('With Y Train (log-log scale)', fontsize=18)
axes[0,2].set_title('With Y Test', fontsize=18)
axes[0,3].set_title('With Y Test (log-log scale)', fontsize=18)

axes[0,0].set_xlabel('Redshift (Truth)', fontsize=18)
axes[0,1].set_xlabel('Redshift (Truth)', fontsize=18)
axes[0,2].set_xlabel('Redshift (Truth)', fontsize=18)
axes[0,3].set_xlabel('Redshift (Truth)', fontsize=18)

axes[0,0].set_ylabel('Redshift (Photometric)', fontsize=18)

axes[1,0].set_title('Without Y Train', fontsize=18)
axes[1,1].set_title('Without Y Train (log-log scale)', fontsize=18)
axes[1,2].set_title('Without Y Test', fontsize=18)
axes[1,3].set_title('Without Y Test (log-log scale)', fontsize=18)

axes[1,0].set_xlabel('Redshift (Truth)', fontsize=18)
axes[1,1].set_xlabel('Redshift (Truth)', fontsize=18)
axes[1,2].set_xlabel('Redshift (Truth)', fontsize=18)
axes[1,3].set_xlabel('Redshift (Truth)', fontsize=18)

axes[1,0].set_ylabel('Redshift (Photometric)', fontsize=18)
fig.tight_layout()
plt.show()

def plot_heat_map(pis, alphas, betas, ys, num_seg, color_map, ax):
  count=0
  # devide Y values in num_seg segments
  segment=(np.max(ys)-np.min(ys))/num_seg
  # store index of non ordered Ys
  index_y=np.arange(len(ys))
  # array to store indeces of "equispaced" Y values, negative if no value
  i_equi=-np.ones(num_seg)
  for i in range(num_seg):
    # select lower and upper limits of i segment
    z_l=count*segment+np.min(ys)
    count += 1
    z_u=count*segment+np.min(ys)
    # select indeces of sources with Y values between z_l and z_u
    i_trun=index_y[(ys>z_l).flatten() & (ys<z_u).flatten()]
    # if this array is not empty, select one of those randomly
    if i_trun.size != 0:
      i_equi[i]=int(i_trun[np.random.randint(0,len(i_trun),1)])
  i_equi=i_equi.astype(int)

  # initialize heat-map
  heat = np.zeros((num_seg,num_seg))
  # For each of those sources, plot the PDF
  y_ph_z = np.linspace(0.0, np.max(ys), num_seg)
  for j in range(num_seg):
    if i_equi[j]!=-1: # if the array is not empty
      final = np.zeros_like(y_ph_z)
      for i, (weight_mix, alpha_mix, beta_mix) in enumerate(zip(pis[i_equi[j],:], alphas[i_equi[j],:], betas[i_equi[j],:])):
        dist = tfd.Gamma(concentration=alpha_mix, rate=beta_mix)
        pdf = dist.prob(y_ph_z)
        temp = pdf.numpy() * weight_mix
        final = final + temp
      heat[:,j] = final/np.max(final)
    else:
      heat[:,j] = np.zeros_like(y_ph_z)
      
  ax.imshow(heat,cmap=color_map,origin='lower',extent=[0,max(ys),0,max(ys)])
  ax.set_xticks(np.arange(0,4.5,0.5))
  ax.set_yticks(np.arange(0,4.5,0.5))
  ax.plot([0, max(ys)], [0, max(ys)], linewidth=1, color='k')


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), constrained_layout=True)
np.random.seed(1004)
num_seg=4000

## log-likelihood K=1
plot_heat_map(pi_ts_w,alpha_ts_w,beta_ts_w,Y_test, num_seg, 'Oranges', axes[0])

## log-likelihood K=5
plot_heat_map(pi_ts_wo,alpha_ts_wo,beta_ts_wo,Y_test, num_seg, 'Oranges', axes[1])

axes[0].set_title('$\gamma$-MDN with Y channel', fontsize=14, style='italic')
axes[1].set_title('$\gamma$-MDN without Y channel' , fontsize=14, style='italic')

axes[0].set_xlabel('Redshift (Truth)', fontsize=15)
axes[1].set_xlabel('Redshift (Truth)', fontsize=15)

axes[0].set_ylabel('Redshift (Photometric)', fontsize=15)
axes[1].set_ylabel('Redshift (Photometric)', fontsize=15)

axes[0].tick_params(axis='y', labelsize=13)
axes[0].tick_params(axis='x', labelsize=13)
axes[1].tick_params(axis='y', labelsize=13)
axes[1].tick_params(axis='x', labelsize=13)

#plt.savefig("PDFs_heatmaps_wT.pdf")
plt.show()

steps = [('scaler', RobustScaler()), ('pca', PCA())]

# Reinitiate the transformer pipeline
steps_w = [('scaler', RobustScaler()), ('pca', PCA())]

steps_wo = [('scaler', RobustScaler()), ('pca', PCA())]

pipeline_w = Pipeline(steps_w)

pipeline_wo = Pipeline(steps_wo)



checkpoint_path = "/content/drive/MyDrive/PDFphotoz/aure_original_final_callback_withY.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)

# transform the features
Xtrain_w = pipeline_w.fit_transform(Xw)

# Initialize and compile model (do not comput RI, it crashes)
model_w = Gamma_MDN(K, n_hidden_1, gm_ll_loss, Xtrain_w.shape[1], True)

# fit the model with the optimal number of epochs

n_epoch=850

with tf.device('/device:GPU:0'):
    history_w=model_w.fit(Xtrain_w, yw, 
                      epochs=n_epoch,
                      batch_size = batch_size,
                      verbose=1,
                      callbacks=[cp_callback])

checkpoint_path = "/content/drive/MyDrive/PDFphotoz/aure_original_final_callback_noY.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)

# transform the features
Xtrain_wo = pipeline_wo.fit_transform(Xwo)

# Initialize and compile model (do not comput RI, it crashes)
model_wo = Gamma_MDN(K, n_hidden_1, gm_ll_loss, Xtrain_wo.shape[1], True)

# fit the model with the optimal number of epochs

n_epoch=850

with tf.device('/device:GPU:0'):
    history_wo=model_wo.fit(Xtrain_wo, ywo, 
                      epochs=n_epoch,
                      batch_size = batch_size,
                      verbose=1,
                      callbacks=[cp_callback])

feat_sel = ['COADD_OBJECT_ID','Z','SOF_CM_T','SOF_CM_MAG_CORRECTED_G', 'SOF_CM_MAG_CORRECTED_R', 'SOF_CM_MAG_CORRECTED_I', 'SOF_CM_MAG_CORRECTED_Z', 'MAG_AUTO_CORRECTED_Y', 'ERR_Z', 'SOF_CM_T_ERR','SOF_CM_MAG_ERR_G', 'SOF_CM_MAG_ERR_R', 'SOF_CM_MAG_ERR_I', 'SOF_CM_MAG_ERR_Z', 'MAGERR_AUTO_Y', 'ZMEAN_Y4']
df_test = test_data_set[feat_sel]

### Remove galaxies with magnitude errors < 0 and small redshifts (stars)
df_test = df_test[(df_test.Z<2) &
                    (df_test.SOF_CM_MAG_ERR_G>0.) &
                    (df_test.SOF_CM_MAG_ERR_R>0.) &
                    (df_test.SOF_CM_MAG_ERR_I>0.) &
                    (df_test.SOF_CM_MAG_ERR_Z>0.) &
                    (df_test.MAG_AUTO_CORRECTED_Y>0.) &
                    (df_test.MAG_AUTO_CORRECTED_Y<40.) &
                    (df_test.SOF_CM_T>0.)]

# let's change the column names for convenience
feat_names = {'COADD_OBJECT_ID': 'id','Z': 'redshift','SOF_CM_T': 'T','SOF_CM_MAG_CORRECTED_G': 'g', 'SOF_CM_MAG_CORRECTED_R': 'r', 'SOF_CM_MAG_CORRECTED_I': 'i', 'SOF_CM_MAG_CORRECTED_Z': 'z', 'MAG_AUTO_CORRECTED_Y': 'Y', 'ERR_Z': 'redshift_err', 'SOF_CM_MAG_ERR_G': 'g_err', 'SOF_CM_MAG_ERR_R': 'r_err', 'SOF_CM_MAG_ERR_I': 'i_err', 'SOF_CM_MAG_ERR_Z': 'z_err', 'MAGERR_AUTO_Y': 'Y_err', 'ZMEAN_Y4': 'ZMEAN_Y4'}
df_test=df_test.rename(columns=feat_names)


df_test_aug = df_test[['id','ZMEAN_Y4','redshift','T','g','r','i','z','Y']].copy()
col_names = ['g','r','i','z','Y']
for i in range(len(col_names)-1):
  for j in range(i+1,len(col_names)):
      df_test_aug[col_names[i]+'-'+col_names[j]]= df_test_aug[col_names[i]]-df_test_aug[col_names[j]]

# take log10 of magnitude errors
feats = ['T']
df_test_aug[feats]=np.log10(df_test_aug[feats])

# dataframe with Y channel (w)
#df_test_wY = df_test_aug.copy()

# dataframe without Y channel (woY)
df_test_woY = df_test_aug.drop(['Y','g-Y','r-Y','i-Y','z-Y'], axis=1)

### The training features with Y channel ###
#feat_test = list(df_test_wY)
#X_ts_w = df_test_wY[feat_test[3:]].to_numpy()

### The training features without Y channel ###
feat_test = list(df_test_woY)
X_ts_wo = df_test_woY[feat_test[3:]].to_numpy()

### Target variable and DNF prediction
y_ts = df_test_woY.redshift.to_numpy()
y_ts = y_ts[:,np.newaxis]
y_dnf = df_test.ZMEAN_Y4.to_numpy()
y_dnf = y_dnf[:,np.newaxis]


## with Y channel

#output_w=model_w.predict(pipeline_w.transform(X_ts_w))
#pi_ts_w, alpha_ts_w, beta_ts_w = get_mixture_coef_np(output_w)

## Without Y channel

output_wo=model_wo.predict(pipeline_wo.transform(X_ts_wo))
pi_ts_wo, alpha_ts_wo, beta_ts_wo = get_mixture_coef_np(output_wo)

def return_df_bis(pis, alphas, betas, Y_valid, Y_dnf):
    """
    Given the output of the MDN, returns
    a DataFrame with mean, variance and stddev added
    """
    pi_names = ['pi_' + str(i) for i in range(len(pis[0]))]
    alpha_names = ['alpha_' + str(i) for i in range(len(pis[0]))]
    beta_names = ['beta_' + str(i) for i in range(len(pis[0]))]
    means_names = ['mean_' + str(i) for i in range(len(pis[0]))]
    std_names = ['sdtdev_' + str(i) for i in range(len(pis[0]))]
    names = pi_names + alpha_names + beta_names + means_names + std_names
    temp = np.concatenate((pis, alphas, betas, alphas/betas, np.sqrt(alphas)/betas), axis=1)
    df = pd.DataFrame(temp, columns=names)
    
    variances = alphas/betas**2
    means = (alphas / betas)
    modes = pdf_mode(pis,alphas,betas)
        
    df['Mode'] = modes    
    df['Mean'] = np.average(means, weights=pis, axis=1)
    df['variance'] =  np.average(means**2 + variances**2, weights=pis, axis=1) - df['Mean'].values**2
    df['stddev'] = np.sqrt(df.variance)
    df['redshift'] = Y_valid
    df['ZMEAN_Y4'] = Y_dnf
    return df

def plot_gamma_mix(pis, alphas, betas):
    """
    Plots the mixture of gamma models to axis=ax
    """
    x = np.linspace(0.0, 4, 600)
    final = np.zeros_like(x)
    for i, (weight_mix, alpha_mix, beta_mix) in enumerate(zip(pis, alphas, betas)):
        dist = tfd.Gamma(concentration=alpha_mix, rate=beta_mix)
        pdf = dist.prob(x)
        temp = pdf.numpy() * weight_mix
        final = final + temp
    return x,final
    #ax.plot(x, final, c=color)

## With Y channel
#res_ts_w = return_df_bis(pi_ts_w, alpha_ts_w, beta_ts_w, y_ts,y_dnf)

## Without Y channel
res_ts_wo = return_df_bis(pi_ts_wo, alpha_ts_wo, beta_ts_wo, y_ts,y_dnf)


print(pi_ts_wo.shape, alpha_ts_wo.shape, beta_ts_wo.shape)
print(pi_ts_wo[0])
#res_ts_wo_pdf = plot_gamma_mix(pi_ts_wo, alpha_ts_wo, beta_ts_wo)
pdf_file = open('pdfs_save.txt','w')
pdf_file.write('#pi_1 pi_2 pi_3 pi_4 pi_5 alpha_0 alpha_1 alpha_2 alpha_3 alpha_4 alpha_5 beta_0 beta_1 beta_2 beta_3 beta_4 beta_5\n')
for i in range(238639):
  pdf_file.write('%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n'%(pi_ts_wo[i][0], pi_ts_wo[i][1], pi_ts_wo[i][2], pi_ts_wo[i][3], pi_ts_wo[i][4], alpha_ts_wo[i][0], alpha_ts_wo[i][1], alpha_ts_wo[i][2], alpha_ts_wo[i][3], alpha_ts_wo[i][4], beta_ts_wo[i][0], beta_ts_wo[i][1], beta_ts_wo[i][2], beta_ts_wo[i][3], beta_ts_wo[i][4]))
pdf_file.close()

amount=1000
plt.scatter(res_ts_wo.redshift.iloc[:amount],res_ts_wo.ZMEAN_Y4.iloc[:amount])
plt.scatter(res_ts_wo.redshift.iloc[:amount],res_ts_wo.Mean.iloc[:amount])
#plt.scatter(res_ts_w.redshift.iloc[:amount],res_ts_w.Mean.iloc[:amount])
plt.show()

#F_t_test_w=PIT(y_ts,output_w)
#RI_test_w=RI_metric(y_ts,output_w).numpy()

F_t_test_wo=PIT(y_ts,output_wo)
RI_test_wo=RI_metric(y_ts,output_wo).numpy()

fig = plt.figure(figsize=(13, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

n_bins=20
#sns.distplot(F_t_test_w.numpy(), hist=True, kde=False, 
#             bins=n_bins, color = 'orange',
#             hist_kws={'edgecolor':'black'}, ax=ax1)

n_bins=20
sns.distplot(F_t_test_wo.numpy(), hist=True, kde=False, 
             bins=n_bins, color = 'blue',
             hist_kws={'edgecolor':'black'}, ax=ax2)

# Perfectly callibrated fit together with 90% CI
ax1.axhline(y=len(y_ts)/n_bins, linestyle='--', color='black', alpha=1)
ax1.axhline(y=0.95*len(y_ts)/n_bins, linestyle='-.', color='black', alpha=1)
ax1.axhline(y=1.05*len(y_ts)/n_bins, linestyle='-.', color='black', alpha=1)
ax1.axvline(x=0.5,linestyle='--', color='black', alpha=1)
ax2.axhline(y=len(y_ts)/n_bins, linestyle='--', color='black', alpha=1)
ax2.axhline(y=0.95*len(y_ts)/n_bins, linestyle='-.', color='black', alpha=1)
ax2.axhline(y=1.05*len(y_ts)/n_bins, linestyle='-.', color='black', alpha=1)
ax2.axvline(x=0.5,linestyle='--', color='black', alpha=1)

ax1.set_xlabel('Probability Integral Transform', fontsize=14)
#ax1.set_title('With Y channel Validation (RI = %.4f)' %(RI_test_w), fontsize=15)
ax1.set_xlabel('Probability Integral Transform', fontsize=14)
ax2.set_title('Without Y channel Validation (RI = %.4f)' %(RI_test_wo), fontsize=15)
ax1.set_ylabel('Frequency', fontsize=14)
ax2.set_xlabel('Probability Integral Transform', fontsize=14)
ax2.set_ylabel('Frequency', fontsize=14)

plt.tight_layout()

plt.show()

redshift_bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

redshift_bins= np.arange(0.1,1.4,0.1)

#redshift_bins =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#photo_z_w = res_ts_w['Mean'].to_numpy()

photo_z_wo = res_ts_wo['Mean'].to_numpy()

dimension = len(redshift_bins)-1

z_rel = np.zeros(dimension)

RI_w = np.zeros(dimension)

RI_wo = np.zeros(dimension)

fig, ax = plt.subplots(2,dimension,figsize=(25,6))


for i1 in range(dimension):
      
      z_rel[i1] = (redshift_bins[i1]+redshift_bins[i1+1])/2.
      
      #zmask_w = (photo_z_w >= redshift_bins[i1])*(photo_z_w < redshift_bins[i1+1])
      #zmask_w = zmask_w.flatten()
      zmask_wo = (photo_z_wo >= redshift_bins[i1])*(photo_z_wo < redshift_bins[i1+1])
      zmask_wo = zmask_wo.flatten()
      #temp_output_w = output_w[zmask_w,:]
      temp_output_wo = output_wo[zmask_wo,:]
      #temp_y_w = y_ts[zmask_w]
      temp_y_wo = y_ts[zmask_wo]

      #RI_w[i1]=RI_metric(temp_y_w,temp_output_w).numpy()
      RI_wo[i1]=RI_metric(temp_y_wo,temp_output_wo).numpy()
      
      #PIT_w=PIT(temp_y_w,temp_output_w).numpy()
      PIT_wo=PIT(temp_y_wo,temp_output_wo).numpy()
      
      n_bins=20
      #sns.histplot(PIT_w, 
      #            stat = 'density',
      #            bins=n_bins, color = 'coral',
      #            fill = False,
      #            legend = False,
      #            ax=ax[0,i1])
      
      n_bins=20
      sns.histplot(PIT_wo, 
                  stat = 'density',
                  bins=n_bins, color = 'tag:blue',
                  fill = False,
                  legend = False,
                  ax=ax[1,i1])
      
      ax[0,i1].set_title('z_rel = %.2f' %(z_rel[i1]), fontsize=10)
      
plt.tight_layout()
plt.show()

#plt.plot(z_rel,RI_w, c='coral')
plt.plot(z_rel,RI_wo, c='blue')
plt.ylabel('RI')
plt.xlabel('Redshift')
plt.legend(['Without Y channel'])#'With Y channel',
plt.show()


df_means_pred = res_ts_wo[['Mean']].copy()

df_means_pred = df_means_pred.rename(columns={'Mean': 'z_gammaMDN_woY'})
#df_means_pred['z_gammaMDN_woY'] = res_ts_wo['Mean']
df_means_pred['z_dnf'] = res_ts_wo['ZMEAN_Y4']
df_means_pred['z_spec'] = res_ts_wo['redshift']

df_means_pred.head(5)




fig, ax = plt.subplots(1,1,figsize=(7,7))
ax.scatter(df_means_pred.z_spec,df_means_pred.z_dnf,s=0.5)
#ax.scatter(df_means_pred.z_spec,df_means_pred.z_gammaMDN_wY,s=0.5)
ax.scatter(df_means_pred.z_spec,df_means_pred.z_gammaMDN_woY,s=0.5)
ax.plot([0, max(df_means_pred.z_spec)], [0, max(df_means_pred.z_spec)], linewidth=1, color='k')
ax.set_xlim(0, max(df_means_pred.z_spec))
ax.set_ylim(0, max(df_means_pred.z_spec))
ax.set_xlabel('Redshift (Truth)', fontsize=18)
ax.set_ylabel('Redshift (Photometric)', fontsize=18)
ax.set_title('Test Set', fontsize=18)
plt.show()

# Load the Drive helper and mount

# This will prompt for authorization.
drive.mount('/content/drive')

os.chdir('/content/drive/My Drive/')

#create a csv file
df_means_pred.to_csv("predicted_means_gammaMDN_vs_DNF_vs_actual_My_gammaMDN_withT_1_11Nov2020.csv", index=True, header=True)
