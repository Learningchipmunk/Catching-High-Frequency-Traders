from verstack.stratified_continuous_split import scsplit
from random import shuffle
from math import log10, floor
import pandas as pd
import numpy as np


class AddType():
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.labels = y
        return self

    def transform(self, X, y=None):
        X_ = X

        # Checks if labels are given:
        """This line has been added in order to preprocess the test data. Indeed, we do not have labels for the test data."""
        if(self.labels is not None):
            X_ = pd.merge(X, self.labels, how='left', left_on=[
                'Trader'], right_on=['Trader'])

        return X_


class Inverter():
    def __init__(self, cols=[]):
        self.cols = cols

    def invert(self, x):
        # invert and fill NaN with 0 (for OMR, OTR, OCR)
        if x != x:
            return 0
        else:
            return 1/x

    def fit(self, X, y=None):
        # pre computation
        return self

    def transform(self, X, y=None):
        # transform data
        X_ = X.copy()
        for col in self.cols:
            X_[col] = X_[col].apply(self.invert)
        return X_


class CountSharesSameDay():
    def __init__(self, s='Nber_shares_same_day'):
        self.feature_name = s

    def fit(self, X, y=None):
        # pre computation
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        # count number of shares the same day
        groups = X_.groupby(['Trader', 'Day'], as_index=False).size()
        groups = groups.rename(
            columns={"size": self.feature_name})  # rename col

        X_ = pd.merge(X_, groups, how='left', left_on=[
                      'Trader', 'Day'], right_on=['Trader', 'Day'])

        return X_


class DropCols():
    def __init__(self, cols=[]):
        self.cols = cols

    def fit(self, X, y=None):
        # pre computation
        return self

    def transform(self, X, y=None):
        return X.drop(self.cols, axis=1)


class MeanImput():
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # pre computation
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        for col in X.columns[X.isna().any()].tolist():
            X_[col] = X_[col].fillna(value=X_[col].mean())
        return X_


class DropMixTraders():
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        X_ = X_[X_.type != 'MIX']
        return X_


def match(x):
    HFT = 'HFT'
    NON_HFT = 'NON HFT'
    MIX = 'MIX'
    if x in HFT:
        return 2
    elif x in NON_HFT:
        return 0
    elif x in MIX:
        return 1
    else:
        return -1


def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)


class SplitTrainTest_Gen():
    '''
    X must have a 'type' column
    There can be any number of types
    no Y is necessary
    '''

    def __init__(self, train_size, tol=0.05, folder='Processed_data/', suffix='1', drop_traders=True, drop_type=True):
        '''
        train_size: relative size of train_set (between 0 and 1)
        tol: tolerance with regards to various proportions
        folder: folder where test and train sets will be saved
        suffix: suffix appended to files' names
        '''
        self.train_size = train_size
        self.folder = folder
        self.tol = tol
        self.suffix = suffix
        self.drop_traders = drop_traders
        self.drop_type = drop_type

    def fit(self, X, y=None):

        types_props = X['type'].value_counts()/X.shape[0]
        map_type_traders = {}
        train_sizes = {}
        for t in types_props.index:
            # for each type of trader, gather traders of that type
            map_type_traders[t] = list(set(X[X['type'] == t]['Trader']))
            train_sizes[t] = int(self.train_size * len(map_type_traders[t]))

        target_train_size = self.train_size*X.shape[0]
        count = 0
        while(True):
            self.train_traders = []  # traders of the test set
            for t in types_props.index:
                shuffle(map_type_traders[t])
                self.train_traders += map_type_traders[t][:train_sizes[t]]

            count += 1
            X_train = X[X['Trader'].isin(self.train_traders)]

            if(abs(X_train.shape[0] - target_train_size) < self.tol*target_train_size):
                isOK = True
                train_props = X_train['type'].value_counts()/X_train.shape[0]

                for t in types_props.index:
                    if(abs(types_props[t] - train_props[t]) > self.tol):
                        isOK = False
                        break
                if(isOK):
                    break

        self.test_traders = []
        for t in types_props.index:
            self.test_traders += map_type_traders[t][train_sizes[t]:]

        return self

    def transform(self, X, y=None):
        X_train = X[X['Trader'].isin(self.train_traders)].copy()
        X_test = X[X['Trader'].isin(self.test_traders)].copy()

        y_train = X_train['type'].copy()
        y_test = X_test['type'].copy()

        if(self.drop_type):
            X_train.drop('type', axis=1, inplace=True)
            X_test.drop('type', axis=1, inplace=True)

        print("Train set size:", round_sig(X_train.shape[0]/X.shape[0]))
        p_train = y_train.value_counts()/y_train.shape[0]
        p_test = y_test.value_counts()/y_test.shape[0]

        for t in p_train.index:
            print("y_train proportions "+t, round_sig(p_train[t]))

        print("\nTest set size:", round_sig(X_test.shape[0]/X.shape[0]))
        for t in p_test.index:
            print("y_test proportions "+t, round_sig(p_test[t]))

        # Convert labels str to int
        y_train = y_train.apply(match)
        y_test = y_test.apply(match)

        # Dropping Traders:
        if(self.drop_traders):
            X_train.drop('Trader', axis=1, inplace=True)
            X_test.drop('Trader', axis=1, inplace=True)

        X_train.to_pickle(self.folder+'X_train_'+self.suffix+'.pkl')
        X_test.to_pickle(self.folder+'X_test_'+self.suffix+'.pkl')

        y_train.to_pickle(self.folder+'y_train_'+self.suffix+'.pkl')
        y_test.to_pickle(self.folder+'y_test_'+self.suffix+'.pkl')

        return X_train, y_train


class SplitTrainTest_Default():

    def __init__(self, train_size, folder='Processed_data/', suffix='1'):
        self.train_size = train_size
        self.folder = folder
        self.suffix = suffix

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.drop
        X_train, X_test, y_train, y_test = scsplit(X.drop('type', axis=1),
                                                   X['type'],
                                                   stratify=X['type'],
                                                   train_size=self.train_size,
                                                   test_size=1-self.train_size,
                                                   continuous=False)

        print("Train set size:", round_sig(X_train.shape[0]/X.shape[0]))

        unique, counts = np.unique(y_train, return_counts=True)
        nums = dict(zip(unique, counts))
        for t in list(set(X['type'])):
            print("y_train proportions "+t,
                  round_sig(nums[t]/X_train.shape[0]))

        print("\nTest set size:", round_sig(X_test.shape[0]/X.shape[0]))

        unique, counts = np.unique(y_test, return_counts=True)
        nums = dict(zip(unique, counts))
        for t in list(set(X['type'])):
            print("y_test proportions "+t, round_sig(nums[t]/X_test.shape[0]))

        with open(self.folder+'X_train_'+self.suffix+'.npy', 'wb') as f:
            np.save(f, X_train)

        with open(self.folder+'X_test_'+self.suffix+'.npy', 'wb') as f:
            np.save(f, X_test)

        with open(self.folder+'y_train_'+self.suffix+'.npy', 'wb') as f:
            np.save(f, y_train)

        with open(self.folder+'y_test_'+self.suffix+'.npy', 'wb') as f:
            np.save(f, y_test)

        return X_train, y_train
