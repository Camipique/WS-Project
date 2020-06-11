# -*- coding: utf-8 -*-
"""
@author: Carlos
"""

import time

import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def process_criteo():
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
    
    print("Saving processed criteo data...\n")
    data_criteo.to_csv("./data/criteo/criteo_sample_processed.csv")
    
    print("Creating train and test subsets on criteo data...\n")
    split_dataset(data_criteo, "criteo", 0.5)
    
    return data_criteo
    
def process_avazu():
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
    
    print("Saving processed avazu data...\n")
    data_avazu.to_csv("./data/avazu/avazu_sample_processed.csv")
    
    print("Creating train and test subsets on avazu data...\n")
    split_dataset(data_avazu, "avazu", 0.5)
    
    return data_avazu

"""
    dataset: dataset to split into train and test subsets
    dataset_name: dataset name
    test_proportion: proportion of the test subset. default is 20% of the size of the set
"""
def split_dataset(dataset, dataset_name, test_proportion = 0.2):
    train, test = train_test_split(dataset, test_size=test_proportion, stratify = dataset["label"])
    
    train.to_csv("./data/{name}/train_{size}_{name}.csv".format(name = dataset_name, size = 1 - test_proportion))
    test.to_csv("./data/{name}/test_{size}_{name}.csv".format(name = dataset_name, size = test_proportion))
    
    
if __name__ == "__main__":
    start_time = time.time()
    
    data_criteo = process_criteo()
    data_avazu = process_avazu()
        
    print("Program ended in {time} seconds.".format(time = round(time.time() - start_time, 2)))