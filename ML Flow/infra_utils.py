import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.utils import shuffle

import mlflow
import os

def get_raw_data(df):
    """
    """
    
    # data_file = ('dataset' +  os.path.sep +  'creditcard.csv')
    # df = pd.read_csv(data_file)

    df = shuffle(df, random_state=101)
    df_0 = df.loc[df['Class'] == 0].iloc[0 : int(len(df) / 100)]
    df_1 = df[df['Class'] == 1]
    df = pd.concat([df_0, df_1])

    X_train, X_test, y_train, y_test = train_test_split(df.drop('Class', axis=1), df['Class'], test_size=0.33, random_state=42)

    X_train['target'] = y_train
    X_test['target'] = y_test
    df_train = X_train.copy()
    df_test = X_test.copy()

    return df_train, df_test






