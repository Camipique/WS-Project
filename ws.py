# -*- coding: utf-8 -*-
"""
@author: Carlos
"""
import time

import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# To read, pre-process and split datasets
PROCESS = False
# Proportion of the test subset
TEST_PROPORTION = 0.5

def process_criteo():
    
    if PROCESS:
        print("Reading criteo data...")
        
        data_criteo = pd.read_csv('./data/criteo/criteo_sample.csv').iloc[:, 1:]
        
        print("Processing criteo data...")
        
        features_labels = data_criteo.columns
        
        sparse_features_labels = features_labels[14:]
        dense_features_labels = features_labels[1:14]
    
        data_criteo[sparse_features_labels] = data_criteo[sparse_features_labels].fillna('None', None)
        data_criteo[dense_features_labels] = data_criteo[dense_features_labels].fillna(0, None)
        
        for feat in sparse_features_labels:
            lbe = LabelEncoder()
            data_criteo[feat] = lbe.fit_transform(data_criteo[feat])
        
        mms = MinMaxScaler(feature_range=(0, 1))
        data_criteo[dense_features_labels] = mms.fit_transform(data_criteo[dense_features_labels])
        
        print("Saving processed criteo data...")
        data_criteo.to_csv("./data/criteo/criteo_sample_processed.csv")
        
        train_criteo, test_criteo = split_dataset(data_criteo, "criteo")
    else:
        print("Reading processed criteo data...")
        data_criteo = pd.read_csv('./data/criteo/criteo_sample_processed.csv').iloc[:, 1:]
        train_criteo = pd.read_csv("./data/criteo/train_{size}_criteo.csv".format(size = TEST_PROPORTION)).iloc[:, 1:]
        test_criteo = pd.read_csv("./data/criteo/test_{size}_criteo.csv".format(size = TEST_PROPORTION)).iloc[:, 1:]
        
    
    return data_criteo, train_criteo, test_criteo
    
def process_avazu():
    if PROCESS:
        print("Reading avazu data...")
        
        data_avazu = pd.read_csv('./data/avazu/avazu_sample.csv')
        
        print("Processing avazu data...")
        # Dropping "hour" column since the value is always the same
        data_avazu.drop("hour", axis=1, inplace=True)
        
        # Swapping id and label columns
        cols = list(data_avazu)
        cols[0], cols[1] = cols[1], cols[0]
        data_avazu = data_avazu.loc[:, cols]
        
        features_labels = data_avazu.columns
        
        sparse_features_labels = features_labels[1:]
        
        for feat in sparse_features_labels:
            lbe = LabelEncoder()
            data_avazu[feat] = lbe.fit_transform(data_avazu[feat])
        
        print("Saving processed avazu data...")
        data_avazu.to_csv("./data/avazu/avazu_sample_processed.csv")
        
        train_avazu, test_avazu = split_dataset(data_avazu, "avazu")
    else:
        print("Reading processed avazu data...")
        data_avazu = pd.read_csv('./data/avazu/avazu_sample_processed.csv').iloc[:, 1:]
        train_avazu = pd.read_csv("./data/avazu/train_{size}_avazu.csv".format(size = TEST_PROPORTION)).iloc[:, 1:]
        test_avazu = pd.read_csv("./data/avazu/test_{size}_avazu.csv".format(size = TEST_PROPORTION)).iloc[:, 1:]
        
        
    return data_avazu, train_avazu, test_avazu


def split_dataset(dataset, dataset_name):
    print("Creating train and test subsets on {name} data...\n".format(name = dataset_name))
    
    train, test = train_test_split(dataset, test_size=TEST_PROPORTION, stratify = dataset["label"])
    
    train.to_csv("./data/{name}/train_{size}_{name}.csv".format(name = dataset_name, size = TEST_PROPORTION))
    test.to_csv("./data/{name}/test_{size}_{name}.csv".format(name = dataset_name, size = TEST_PROPORTION))
        
    return train, test


def test_PNN_criteo():
    pass

def test_FNN():
    pass

def test_DFM():
    pass
    
    
if __name__ == "__main__":
    
    start_time = time.time()
    
    data_criteo, train_criteo, test_criteo = process_criteo()
    data_avazu, train_avazu, test_avazu = process_avazu()
    
        
    print("Program ended in {time} seconds.".format(time = round(time.time() - start_time, 2)))