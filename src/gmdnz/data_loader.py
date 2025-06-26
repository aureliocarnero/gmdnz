import pandas as pd
from astropy.table import Table
from sklearn.model_selection import train_test_split

class PhotoZDataLoader:
    def __init__(self, config):
        self.config = config
        self.readnames()
        self.load_data()
        #self.prepare_features_and_target()

        #self.apply_quality_cuts()

    def load_data(self):

        trainsample = self.config.get('data', 'trainsample', fallback='').strip()
        if trainsample:
            self.exist_train = True
            self.train = Table.read(trainsample).to_pandas()
        else:
            self.exist_train = False
            self.train = None

        catalog_path = self.config.get('data', 'galaxysample')
        self.data = Table.read(catalog_path).to_pandas()

    def readnames(self):
        self.dictionary_names = {}
        options = self.config.list_variables('columns')
        for option in options:
            temp = self.config.get('columns', option, dtype=list)
            for t in temp:
                thisval = t.split(':')
                self.dictionary_names[thisval[1]] = thisval[0]

    def print_names(self):
        print(self.dictionary_names)
    '''
    def prepare_features_and_target(self):


        bands = self.config.get('FEATURES', 'bands', dtype=list)
        colors = self.config.get('FEATURES', 'colors', dtype=list)
        
        star_galaxy = self.config.getboolean('FEATURES', 'use_star_galaxy')

        target = 'redshift' #self.config.get('TARGET', 'redshift')

        feature_cols = bands + colors
        
        if star_galaxy:
            feature_cols.append(star_galaxy)

        self.X = self.data[feature_cols].values
        self.y = self.data[target].values

        if self.exist_train:
            self.Xtrain = self.train[feature_cols].values
            self.ytrain = self.train[target].values




    def apply_quality_cuts(self):
        if self.config.get('QUALITY', 'positive_mags', fallback='False') == 'True':
            bands = self.config.get('FEATURES', 'bands', dtype=list)
            for band in bands:
                self.data = self.data[self.data[band] > 0]

        if self.config.get('QUALITY', 'non_nan', fallback='False') == 'True':
            self.data = self.data.dropna()
    '''


    '''
    def get_splits(self, test_size=0.2, val_size=0.1, random_state=42):
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size=(test_size + val_size), random_state=random_state)

        val_fraction = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=random_state)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    '''
