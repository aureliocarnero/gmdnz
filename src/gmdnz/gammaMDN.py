import numpy as np
import tensorflow as tf
import keras
from tensorflow.random import set_seed
set_seed(314)
import pandas as pd
from astropy.table import Table


import seaborn as sns
import matplotlib.pyplot as plt

import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import gamma_mdn as GM
import plotting as PL
import read_config as conf
import data_preparation as DP

#lr = config.getfloat('model', 'learning_rate')
#epochs = config.getint('model', 'epochs')
#input_path = config.get('data', 'input_path')

config = conf.Config(config_file="config_w.ini")

# summarize the model
GM.summary_model()


dat = Table.read(config.get("data", "trainsample"), format='fits')
train_data_set = dat.to_pandas()

dat = Table.read(config.get("data", "galaxysample"), format='fits')
test_data_set = dat.to_pandas()

# Datasets are now stored in a Pandas Dataframe
train_data_set.head()


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
  


#features to plot. Magnitudes
feat_disp =['redshift', 'T', 'g', 'r', 'i', 'z', 'Y']
PL.plot_variables(df_train_aug, feat_disp, namesave = 'plot_variables_mags')

#features to plot. Color differences
feat_disp =['g-r', 'g-i', 'g-z', 'g-Y', 'r-i']
PL.plot_variables(df_train_aug, feat_disp, namesave = 'plot_variables_colors1')

#features to plot. Color differences
feat_disp =['r-z', 'r-Y', 'i-z', 'i-Y', 'z-Y']
PL.plot_variables(df_train_aug, feat_disp, namesave = 'plot_variables_colors2')



### The training features ###
feat_disp = ['T', 'g', 'r', 'i', 'z', 'Y', 'g-r', 'g-i', 'g-z', 'g-Y', 'r-i', 'r-z', 'r-Y', 'i-z', 'i-Y', 'z-Y']

PL.mypairplot(df_train_aug, feat_disp, size = 3000, namesave = 'mypairplot1')


feat_disp = ['redshift', 'T', 'g', 'r', 'i', 'z', 'Y', 'g-r', 'g-i', 'g-z', 'g-Y', 'r-i', 'r-z', 'r-Y', 'i-z', 'i-Y', 'z-Y']
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

X_pca_wo, eigenvec, pipeline_wo = DP.doPCA(Xwo)

PL.pca_plot(X_pca_wo, size=600, namesave = 'pca_plot', eigen_data = [eigenvec, df_train_woY])

var_wo, cum_var_wo = DP.makePCA_table(X_pca_wo, pipeline_wo)


#d_w={"x": np.arange(1,X_pca_w.shape[1]+1),"Var": var_w,"Cum_var": cum_var_w}
#df_pc_w = pd.DataFrame(data=d_w)

d_wo = {"x": np.arange(1, X_pca_wo.shape[1] + 1), "Var": var_wo, "Cum_var": cum_var_wo}
df_pc_wo = pd.DataFrame(data = d_wo)


#PL.pca_var(df_pc_w, namesave = 'pca_var_w', title = 'With Y channel')
PL.pca_var(df_pc_wo, namesave = 'pca_var_wo', title = 'Without Y channel')


temp_df = pd.DataFrame(X_pca_wo, columns = ['PC1','PC2','PC3','PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11']) #, 'PC12', 'PC13', 'PC14', 'PC15', 'PC16'])

PL.mypairplot(temp_df, ['PC1','PC2','PC3','PC4'], size = 5000, namesave = 'mypairplot2')


# remove column with redshift error (let Y changel)
#df_train_wY = df_train_aug.drop(['redshift_err'], axis=1)
df_train_wY = df_train_aug.copy()

# dataframe without Y channel (woY)
#df_train_woY = df_train_aug.drop(['redshift_err','Y','Y_err','g-Y','g-Y_err','r-Y','r-Y_err','i-Y','i-Y_err','z-Y','z-Y_err'], axis=1)
df_train_woY = df_train_aug.drop(['Y','g-Y','r-Y','i-Y','z-Y'], axis=1)


### The training features with Y channel ###
feat_train = list(df_train_woY)

Xwo = df_train_woY[feat_train[2:]].to_numpy()
ywo = df_train_woY.redshift


print(Xwo.shape)



# parameters to initiate and train the gamma-MDN

# number of mixtures
K = 5
# number of neurons for first hidden layer
n_hidden_1 = 128
# batch_size
batch_size = 256*2
# number of epochs
n_epoch = 200 # 400 is more than enough
# number of K folds
n_folds = 3
# validation frequency
val_freq = 1


# define training and test partitions
X_training_wo, Xtest_wo, Y_training_wo, Ytest_wo = train_test_split(Xwo, ywo, test_size=15/100, shuffle=True, random_state=1)





'''TEMP
# Define the K-fold Cross Validator
kfold_wo = KFold(n_splits = n_folds, shuffle = True, random_state = 0)

# K-fold Cross Validation model evaluation
fold_no = 1
isFirst = True
for train, val in kfold_wo.split(X_training_wo, Y_training_wo):
 
   # fit and transform the training partition
    #X_train = pipeline_wo.fit_transform(X_training_wo[train])
    Y_train = Y_training_wo.to_numpy()[train, np.newaxis]
    # transform validation partition
    #    X_val = pipeline_wo.transform(X_training_wo[val])
    Y_val = Y_training_wo.to_numpy()[val, np.newaxis]


    # Generate a print
    print('------------------------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} out of {n_folds}')
    # Increase fold number
    fold_no = fold_no + 1


    # evaluate and store performance
    if isFirst:
        X_train, _, pipe = DP.doPCA(X_training_wo[train])
        X_val = DP.doPCA(X_training_wo[val], pipeline = pipe, fit_transform = False)
        history, model, mycp = GM.model_fit(X_train, Y_train, K, n_hidden_1, True, validation_data = (X_val, Y_val), toSave = 'test_in_original_noY_model_callback.weights.h5', inloop = False)
        isFirst = False
        cv_train_loss_wo_cv = history.history['loss']
        cv_val_loss_wo_cv = history.history['val_loss']
        cv_train_RI_wo_cv = history.history['ri_metric']
        cv_val_RI_wo_cv = history.history['val_ri_metric']

    else: 
        X_train, _, pipe = DP.doPCA(X_training_wo[train], pipeline = pipe)
        X_val = DP.doPCA(X_training_wo[val], pipeline = pipe, fit_transform = False)

        history, model, mycp = GM.model_fit(X_train, Y_train, K, n_hidden_1, True, validation_data = (X_val, Y_val), toSave = 'test_in_original_noY_model_callback.weights.h5', inloop = True, mycp = mycp)
        cv_train_loss_wo_cv = np.vstack((cv_train_loss_wo_cv, history.history['loss']))
        cv_val_loss_wo_cv = np.vstack((cv_val_loss_wo_cv, history.history['val_loss']))
        cv_train_RI_wo_cv = np.vstack((cv_train_RI_wo_cv, history.history['ri_metric']))
        cv_val_RI_wo_cv = np.vstack((cv_val_RI_wo_cv, history.history['val_ri_metric']))
    


# epochs arrays
epochs_train = history.epoch + np.ones(len(history.epoch))
epochs_val = history.epoch + val_freq * np.ones(len(history.epoch))

# losses arrays for log-likelihood. K=5
mean_train_loss_wo_cv = np.nanmean(cv_train_loss_wo_cv[1:, :], axis = 0)
mean_val_loss_wo_cv = np.nanmean(cv_val_loss_wo_cv[1:, :], axis = 0)
mean_train_RI_wo_cv = np.nanmean(cv_train_RI_wo_cv[1:, :], axis = 0)
mean_val_RI_wo_cv = np.nanmean(cv_val_RI_wo_cv[1:, :], axis = 0)

# optimal epoch
#epoch_min_w_cv=np.argmin(mean_val_loss_w_cv)+1
#epoch_min_RI_w_cv=np.argmin(mean_val_RI_w_cv)+1
epoch_min_wo_cv = np.argmin(mean_val_loss_wo_cv) + 1
epoch_min_wo_cv_wo = np.argmin(mean_val_RI_wo_cv) + 1


PL.train_val_loss(epochs_train, mean_train_loss_wo_cv, mean_train_RI_wo_cv, title = 'without Y channel', valdata = [epochs_val, mean_val_loss_wo_cv, mean_val_RI_wo_cv])
ENDTEMP'''
#print("For model with Y channel: %0.f" %(np.argmin(mean_val_loss_w_cv)+1))
print("")
#print("For model without Y channel: %0.f" %(np.argmin(mean_val_loss_wo_cv) + 1))

X_train_wo, _, pipe = DP.doPCA(X_training_wo)
X_test_wo, _, _ = DP.doPCA(Xtest_wo, pipeline = pipe, fit_transform = False)

Y_train_wo = Y_training_wo.to_numpy()[:, np.newaxis]  # add extra axis as tensorflow expects this 
Y_test_wo = Ytest_wo.to_numpy()[:, np.newaxis]


load = True
if load:
    model_wo = GM.Gamma_MDN(K, n_hidden_1, GM.gm_ll_loss, X_train_wo.shape[1], True)
    model_wo.load_weights('aure_original_second_callback_noY.weights.h5')
else:

    history, model_wo, mycp = GM.model_fit(X_train_wo, Y_train_wo, K, n_hidden_1, True, toSave = 'aure_original_second_callback_noY.weights.h5', inloop = False, n_epoch = 200)


### Without Y channel
# compute PIT and RI
# for the train
output = model_wo.predict(X_train_wo)  

F_t_train_wo = GM.PIT(Y_train_wo, output)
RI_train_wo = GM.RI_metric(Y_train_wo, output).numpy()

# for the test
output = model_wo.predict(X_test_wo) 
F_t_test_wo = GM.PIT(Y_test_wo, output)
RI_test_wo = GM.RI_metric(Y_test_wo, output).numpy()

PL.pit_plot(F_t_train_wo, RI_train_wo, pit_test = F_t_test_wo, ri_test = RI_test_wo, namesave = 'pit_plot', title = r"$\gamma$-MDN without Y channel")



## Without Y channel
# for the train
output=model_wo.predict(X_train_wo)
pi_tr_wo, alpha_tr_wo, beta_tr_wo = GM.get_mixture_coef(output, tonumpy=True)

# for the test
output=model_wo.predict(X_test_wo)
pi_ts_wo, alpha_ts_wo, beta_ts_wo = GM.get_mixture_coef(output, tonumpy=True)




## CRPS
# for the train
#res_tr_w = GM.return_df(pi_tr_w, alpha_tr_w, beta_tr_w, Y_train_w)
# for the test
#res_ts_w = GM.return_df(pi_ts_w, alpha_ts_w, beta_ts_w, Y_test_w)

## Log-likelihood
# for the train
res_tr_wo = GM.return_df(pi_tr_wo, alpha_tr_wo, beta_tr_wo, Y_train_wo)
# for the test
res_ts_wo = GM.return_df(pi_ts_wo, alpha_ts_wo, beta_ts_wo, Y_test_wo)

res_ts_wo.head(5)


PL.dens_hist(pi_ts_wo.flatten(), r'Mixture weights $\pi$', namesave = 'pi_plot', title = 'Without Y channel')
PL.dens_hist(beta_ts_wo.flatten(), 'Rate beta', namesave = 'beta_plot', title = 'Without Y channel')
PL.dens_hist(alpha_ts_wo.flatten(), 'Concentration alpha', namesave = 'alpha_plot', title = 'Without Y channel')



CoV_wo=res_ts_wo['stddev']/res_ts_wo['Mean']*100

PL.dens_hist(CoV_wo[CoV_wo<20], 'Coefficient of Variation (%)', namesave = 'coeff_var_plot', title = 'Without Y channel')

CoV_wo_trun=CoV_wo[CoV_wo<20].copy()

CoV_wo.shape

CoV_wo_trun.shape


PL.plot_examples_pdfs(Y_test_wo, pi_ts_wo, alpha_ts_wo, beta_ts_wo, num_cols = 6, num_rows = 3, namesave = 'examples_pdfs_1', title = 'Random selection')

## In terms of CRPS
variance_wo=res_ts_wo['variance'].values
i_sort_wo=np.argsort(variance_wo)

PL.plot_examples_pdfs(Y_test_wo, pi_ts_wo, alpha_ts_wo, beta_ts_wo, num_cols = 6, num_rows = 3, namesave = 'examples_pdfs_crps', title = 'Sorted by variance', ordered = i_sort_wo)



# plot all test predictions and 5 times as much train predictions

PL.plot_mean_true(res_ts_wo, Y_test_wo, gamma_pdf_train = res_tr_wo, Y_train = Y_train_wo, namesave = 'plot_mean_true', title = '')



#fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5), constrained_layout = True)
np.random.seed(1004)
num_seg = 4000

## log-likelihood K=1
#axes[0] = PL.plot_heat_map(pi_ts_w, alpha_ts_w, beta_ts_w, Y_test, num_seg, 'Oranges', ax = axes[0])

## log-likelihood K=5
PL.plot_heat_map(pi_ts_wo, alpha_ts_wo, beta_ts_wo, Y_test_wo, num_seg, 'Oranges')





X_train_wo, _, pipe = DP.doPCA(Xwo)
#X_test_wo, _, _ = DP.doPCA(Xtest_wo, pipeline = pipe, fit_transform = False)

#Y_train_wo = Y_training_wo.to_numpy()[:, np.newaxis]  # add extra axis as tensorflow expects this 
#Y_test_wo = Ytest_wo.to_numpy()[:, np.newaxis]

load = False

if load:
    model_wo = GM.Gamma_MDN(K, n_hidden_1, GM.gm_ll_loss, X_train_wo.shape[1], True)
    model_wo.load_weights('aure_new_callback_noY.weights.h5')
else:
    history, model_wo, mycp = GM.model_fit(X_train_wo, ywo, K, n_hidden_1, True, toSave = 'aure_original_final_callback_noY.weights.h5', inloop = False, n_epoch = 850)


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
#y_ts = y_ts[:,np.newaxis]
y_dnf = df_test.ZMEAN_Y4.to_numpy()
#y_dnf = y_dnf[:,np.newaxis]

id_gal = df_test.id
id_gal = id_gal.astype(int)
## Without Y channel

output_wo = model_wo.predict(pipe.transform(X_ts_wo))
pi_ts_wo, alpha_ts_wo, beta_ts_wo = GM.get_mixture_coef(output_wo, tonumpy=True)



## With Y channel
#res_ts_w = return_df_bis(pi_ts_w, alpha_ts_w, beta_ts_w, y_ts,y_dnf)

## Without Y channel
res_ts_wo = GM.return_df(pi_ts_wo, alpha_ts_wo, beta_ts_wo, y_ts, Y_dnf = y_dnf, ide = id_gal)
#print(res_ts_wo['id'])

DP.save_pdfs('pdfs_save.txt', pi_ts_wo, alpha_ts_wo, beta_ts_wo, K, ide = id_gal)

PL.plot_mean_true(res_ts_wo, y_ts, namesave = 'plot_mean_true_compa', title = 'Compared to DNF', other = 'OTHERZ')



F_t_test_wo = GM.PIT(y_ts[:,np.newaxis], output_wo)
RI_test_wo = GM.RI_metric(y_ts[:,np.newaxis], output_wo).numpy()


PL.pit_plot(F_t_test_wo, RI_test_wo, namesave = 'pit_plot_final', title = r"$\gamma$-MDN final")

photo_z_wo = res_ts_wo['Mean'].to_numpy()

z_rel, RI_wo = PL.pit_plot_z(output_wo, photo_z_wo, y_ts, redshift_bins = None, n_bins = 20, namesave = 'pit_plot_z', title = '')

PL.val_with_z(RI_wo, z_rel, valname = 'RI', namesave = 'ri_plot_z', title = 'Without Y channel')


res = DP.save_results('predicted_means_gammaMDN_vs_DNF_vs_actual_My_gammaMDN_withT_test.csv', res_ts_wo, summary = False)


'''df_means_pred = res_ts_wo[['Mean']].copy()

df_means_pred = df_means_pred.rename(columns={'Mean': 'z_gammaMDN_woY'})
#df_means_pred['z_gammaMDN_woY'] = res_ts_wo['Mean']
df_means_pred['z_dnf'] = res_ts_wo['OTHERZ']
df_means_pred['z_spec'] = res_ts_wo['redshift']

df_means_pred.head(5)
'''

PL.plot_mean_true(res, res['redshift'], namesave = 'plot_mean_true_final', title = 'TEST SAMPLE', other = 'Mode')

'''
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
'''
# Load the Drive helper and mount

# This will prompt for authorization.

#create a csv file
#df_means_pred.to_csv("predicted_means_gammaMDN_vs_DNF_vs_actual_My_gammaMDN_withT_1_11Nov2020.csv", index=True, header=True)
