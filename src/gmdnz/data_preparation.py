#import read_config as conf
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def doPCA(X, pipeline = None, fit_transform = True):
    if pipeline is None:
        steps = [('scaler', RobustScaler()), ('pca', PCA())]
        pipeline = Pipeline(steps)
    if fit_transform:
        X_pca = pipeline.fit_transform(X)
    else:
        X_pca = pipeline.transform(X)
    eigenvec = pipeline.named_steps.pca.components_
    return X_pca, eigenvec, pipeline


def makePCA_table(X_pca, pipeline):
    var = pipeline.named_steps.pca.explained_variance_ratio_*100
    cum_var = pipeline.named_steps.pca.explained_variance_ratio_.cumsum()*100
    print("Percentage of variability explained by the PCs:")
    print('            Sample')
    print("           Var       Cum_var")
    for i in range(X_pca.shape[1]):
        ind_pc = i + 1
        print('PC %.0f:    %.5f     %.5f'   %(ind_pc, var[i], cum_var[i]))

    return var, cum_var

def save_pdfs(filename, pi, alpha, beta, K, ide = None):
    print('saving pdfs to file', filename)
    pdf_file = open(filename, 'w')

    print('shape pi, alpha, beta', pi.shape, alpha.shape, beta.shape)

    header_str = '#'
    if ide is not None:
        header_str += 'ID '
    a_temp = ''
    pi_temp = ''
    beta_temp = ''
    for i in range(K):
        a_temp += f'alpha_{i+1} ' 
        pi_temp += f'pi_{i+1} '
        beta_temp += f'beta_{i+1} ' 
    header_str = header_str + pi_temp + a_temp + beta_temp + '\n' 
    pdf_file.write(header_str)

    # Write each row dynamically
    for i in range(len(pi)):

        # Concatenate all elements for this row into a single list
        row_values = list(pi[i]) + list(alpha[i]) + list(beta[i])

        if ide is not None:
            row_str = str(ide.to_numpy()[i]) + ' ' + ' '.join(['%.10f' % val for val in row_values]) + '\n'

        else:
            # Convert to string with desired precision
            row_str = ' '.join(['%.10f' % val for val in row_values]) + '\n'
        pdf_file.write(row_str)

    pdf_file.close()

def save_results(namefile, df, summary = False, nametag = 'z_gammaMDN'):

    print('Saving to file', namefile)
    if summary:
        df_means_pred = df[['Mean']].copy()
        df_means_pred = df_means_pred.rename(columns = {'Mean': nametag})
        df_means_pred['OTHERZ'] = df['OTHERZ']
        df_means_pred['z_spec'] = df['redshift']
        df_means_pred.to_csv(namefile, index = True, header = True)
        
        df = df_means_pred

    

    if df.index.name is None:
        df.index.name = 'some_id'
    
    df.head(5)
    df.to_csv(namefile, index = True, header = True)

    return df

