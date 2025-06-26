#import read_config as conf
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def doPCA(X, pipeline = None, fit_transform = True):

    """
    Perform PCA with robust scaling.

    Parameters
    ----------
    X : np.array or pd.DataFrame
        Input data.
    pipeline : sklearn.Pipeline, optional
        Existing pipeline to reuse (should contain 'scaler' and 'pca').
    fit_transform : bool, optional
        If True, fits and transforms X. If False, only transforms X.

    Returns
    -------
    X_pca : np.array
        Transformed data.
    eigenvec : np.array
        Principal component eigenvectors.
    pipeline : sklearn.Pipeline
        The pipeline used (contains scaler and PCA).
    """

    if pipeline is None:
        steps = [('scaler', RobustScaler()), ('pca', PCA())]
        pipeline = Pipeline(steps)

    if fit_transform:
        X_pca = pipeline.fit_transform(X)
    else:
        X_pca = pipeline.transform(X)

    eigenvec = pipeline.named_steps['pca'].components_
    return X_pca, eigenvec, pipeline


def makePCA_table(X_pca, pipeline):

    """
    Print and return the variance explained by each PCA component.

    Parameters
    ----------
    X_pca : np.array
        PCA-transformed data.
    pipeline : sklearn.Pipeline
        Fitted pipeline containing PCA.

    Returns
    -------
    var : np.array
        Percentage of variance explained by each principal component.
    cum_var : np.array
        Cumulative percentage of variance explained.
    """

    var = pipeline.named_steps['pca'].explained_variance_ratio_ * 100
    cum_var = pipeline.named_steps['pca'].explained_variance_ratio_.cumsum() * 100

    print("Percentage of variability explained by the PCs:")
    print("           Var       Cum_var")
    for i in range(X_pca.shape[1]):
        print(f'PC {i+1}:    {var[i]:.5f}     {cum_var[i]:.5f}')

    return var, cum_var


def save_pdfs(filename, pi, alpha, beta, K, ide = None):

    """
    Save gamma mixture PDF parameters to a text file.

    Parameters
    ----------
    filename : str
        Output file path.
    pi : np.array
        Mixture weights (N x K).
    alpha : np.array
        Alpha parameters (N x K).
    beta : np.array
        Beta parameters (N x K).
    K : int
        Number of mixture components.
    ide : pd.Series or np.array, optional
        Optional ID column to save with each row.
    """

    print('Saving PDFs to file:', filename)
    print('Shapes -> pi:', pi.shape, ', alpha:', alpha.shape, ', beta:', beta.shape)

    with open(filename, 'w') as pdf_file:
        header = '#'
        if ide is not None:
            header += 'ID '

        header += ' '.join([f'pi_{i+1}' for i in range(K)]) + ' '
        header += ' '.join([f'alpha_{i+1}' for i in range(K)]) + ' '
        header += ' '.join([f'beta_{i+1}' for i in range(K)]) + '\n'
        pdf_file.write(header)

        for i in range(len(pi)):
            row_values = list(pi[i]) + list(alpha[i]) + list(beta[i])
            if ide is not None:
                row_str = f'{str(ide.to_numpy()[i])} ' + ' '.join([f'{val:.10f}' for val in row_values]) + '\n'
            else:
                row_str = ' '.join([f'{val:.10f}' for val in row_values]) + '\n'
            pdf_file.write(row_str)


def save_results(namefile, df, summary = False, nametag = 'z_gammaMDN'):

    """
    Save results DataFrame to a CSV file. Can optionally save a summary file with selected columns.

    Parameters
    ----------
    namefile : str
        Output file name.
    df : pd.DataFrame
        DataFrame containing results.
    summary : bool, optional
        If True, saves a simplified file with only the mean predictions and key columns.
    nametag : str, optional
        Column name for the mean prediction in the summary file.

    Returns
    -------
    df : pd.DataFrame
        The saved DataFrame (possibly reduced if summary=True).
    """

    print('Saving to file', namefile)

    if summary:
        df_summary = pd.DataFrame({
            nametag: df['Mean'],
            'OTHERZ': df['OTHERZ'],
            'z_spec': df['redshift']
        })
        df_summary.index = df.index
        df_summary.to_csv(namefile, index = True, header = True)
        return df_summary

    if df.index.name is None:
        df.index.name = 'some_id'
    
    df.to_csv(namefile, index = True, header = True)
    return df
