# -*- coding: utf-8 -*-
"""
@author: Carlos
"""
import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from deepctr.models import FNN, PNN, DeepFM
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names

from metrics import compute_auc, compute_log_loss, compute_rmse

PROCESS = False
TEST_PROPORTION =  0.2
PLOT=True

dnn_activation_list = ["relu", "tanh", "sigmoid"]
dnn_dropout_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
dnn_hidden_units_list = [(100, 100), (200, 200), (300, 300), (400, 400), (500, 500), (600, 600), (700, 700), (800, 800)]

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
        
       # data_avazu = pd.read_csv('./data/avazu_2/train.csv')
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

def test_PNN_criteo(data, train, test):    

    print("\nTesting PNN on criteo dataset...\n")
    
    results_activation_function = {"auc": [], "logloss": [], "rmse": []}
    results_dropout = {"auc": [], "logloss": [], "rmse": []}
    results_number_of_neurons = {"auc": [], "logloss": [], "rmse": []}

    auc = 0
    logloss = 0
    rmse = 0
    
    features_labels = train.columns
        
    sparse_features_labels = features_labels[14:]
    dense_features_labels = features_labels[1:14]
    target_label = features_labels[0]
    
    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4,) for feat in sparse_features_labels] + [DenseFeat(feat, 1,) for feat in dense_features_labels]
    
    feature_names = get_feature_names(dnn_feature_columns)
    
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    
    true_y = test[target_label].values
    
    print("\t\t-- ACTIVATION FUNCTIONS --\t\t")
    for dnn_activation in dnn_activation_list:
        print("\nTesting {dnn_activation}...".format(dnn_activation = dnn_activation))
        
        # model = PNN(dnn_feature_columns, use_inner=False, use_outter=True, dnn_activation = dnn_activation, task='binary')
        model = PNN(dnn_feature_columns, use_inner=True, use_outter=False, dnn_activation = dnn_activation, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_activation_function["auc"].append(auc)
        results_activation_function["logloss"].append(logloss)
        results_activation_function["rmse"].append(rmse)
        
    print("\t\t-- DROPOUT RATES --\t\t")
    for dnn_dropout in dnn_dropout_list:
        print("\nTesting {dnn_dropout}...".format(dnn_dropout = dnn_dropout))
        
        # model = PNN(dnn_feature_columns, use_inner=False, use_outter=True, dnn_dropout = dnn_dropout, task='binary')
        model = PNN(dnn_feature_columns, use_inner=True, use_outter=False, dnn_dropout = dnn_dropout, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_dropout["auc"].append(auc)
        results_dropout["logloss"].append(logloss)
        results_dropout["rmse"].append(rmse)
        
    print("\t\t-- HIDDEN UNITS --\t\t")
    for dnn_hidden_units in dnn_hidden_units_list:
        print("\nTesting {dnn_hidden_units}...".format(dnn_hidden_units = dnn_hidden_units))
        
        # model = PNN(dnn_feature_columns, use_inner=False, use_outter=True, dnn_hidden_units = dnn_hidden_units, task='binary')
        model = PNN(dnn_feature_columns, use_inner=True, use_outter=False, dnn_hidden_units = dnn_hidden_units, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0,validation_split=TEST_PROPORTION )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_number_of_neurons["auc"].append(auc)
        results_number_of_neurons["logloss"].append(logloss)
        results_number_of_neurons["rmse"].append(rmse)
        
    if PLOT:
        # create_plots("OPNN", "criteo", results_activation_function, "Activation Function", "activation_func", dnn_activation_list)
        # create_plots("OPNN", "criteo", results_dropout, "Dropout Rate", "dropout", dnn_dropout_list)
        # create_plots("OPNN", "criteo", results_number_of_neurons, "Number of Neurons per layer", "nr_neurons", dnn_hidden_units_list)
        create_plots("PNN", "criteo", results_activation_function, "Activation Function", "activation_func", dnn_activation_list)
        create_plots("PNN", "criteo", results_dropout, "Dropout Rate", "dropout", dnn_dropout_list)
        create_plots("PNN", "criteo", results_number_of_neurons, "Number of Neurons per layer", "nr_neurons", dnn_hidden_units_list)
        

def test_FNN_criteo(data,train,test):
    
    print("\nTesting FNN on criteo dataset...\n")
    
    results_activation_function = {"auc": [], "logloss": [], "rmse": []}
    results_dropout = {"auc": [], "logloss": [], "rmse": []}
    results_number_of_neurons = {"auc": [], "logloss": [], "rmse": []}

    auc = 0
    logloss = 0
    rmse = 0

    features_labels = train.columns
        
    sparse_features_labels = features_labels[14:]
    dense_features_labels = features_labels[1:14]
    target_label = features_labels[0]
    
    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4,) for feat in sparse_features_labels] + [DenseFeat(feat, 1,) for feat in dense_features_labels]
    linear_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4,) for feat in sparse_features_labels] + [DenseFeat(feat, 1,) for feat in dense_features_labels]
 
    feature_names = get_feature_names(linear_feature_columns+ dnn_feature_columns)
    
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    
    true_y = test[target_label].values
    
    print("\t\t-- ACTIVATION FUNCTIONS --\t\t")
    for dnn_activation in dnn_activation_list:
        print("\nTesting {dnn_activation}...".format(dnn_activation = dnn_activation))
        
        model = FNN(linear_feature_columns, dnn_feature_columns, dnn_activation = dnn_activation, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_activation_function["auc"].append(auc)
        results_activation_function["logloss"].append(logloss)
        results_activation_function["rmse"].append(rmse)
        
    print("\t\t-- DROPOUT RATES --\t\t")
    for dnn_dropout in dnn_dropout_list:
        print("\nTesting {dnn_dropout}...".format(dnn_dropout = dnn_dropout))
        
        model = FNN(linear_feature_columns,dnn_feature_columns, dnn_dropout = dnn_dropout, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_dropout["auc"].append(auc)
        results_dropout["logloss"].append(logloss)
        results_dropout["rmse"].append(rmse)

        
    print("\t\t-- HIDDEN UNITS --\t\t")
    for dnn_hidden_units in dnn_hidden_units_list:
        print("\nTesting {dnn_hidden_units}...".format(dnn_hidden_units = dnn_hidden_units))
        
        model = FNN(linear_feature_columns, dnn_feature_columns, dnn_hidden_units = dnn_hidden_units, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_number_of_neurons["auc"].append(auc)
        results_number_of_neurons["logloss"].append(logloss)
        results_number_of_neurons["rmse"].append(rmse)
    
    if PLOT:
        create_plots("FNN", "criteo", results_activation_function, "Activation Function", "activation_func", dnn_activation_list)
        create_plots("FNN", "criteo", results_dropout, "Dropout Rate", "dropout", dnn_dropout_list)
        create_plots("FNN", "criteo", results_number_of_neurons, "Number of Neurons per layer", "nr_neurons", dnn_hidden_units_list)

def test_DFM_criteo(data,train,test):
    print("\nTesting DFM on criteo dataset...\n")
    
    results_activation_function = {"auc": [], "logloss": [], "rmse": []}
    results_dropout = {"auc": [], "logloss": [], "rmse": []}
    results_number_of_neurons = {"auc": [], "logloss": [], "rmse": []}

    auc = 0
    logloss = 0
    rmse = 0

    features_labels = train.columns
        
    sparse_features_labels = features_labels[14:]
    dense_features_labels = features_labels[1:14]
    target_label = features_labels[0]
    
    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4,) for feat in sparse_features_labels] + [DenseFeat(feat, 1,) for feat in dense_features_labels]
    linear_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4,) for feat in sparse_features_labels] + [DenseFeat(feat, 1,) for feat in dense_features_labels]
 
    feature_names = get_feature_names(linear_feature_columns+ dnn_feature_columns)
    
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    
    true_y = test[target_label].values
    
    print("\t\t-- ACTIVATION FUNCTIONS --\t\t")
    for dnn_activation in dnn_activation_list:
        print("\nTesting {dnn_activation}...".format(dnn_activation = dnn_activation))
        
        model = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_activation = dnn_activation, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_activation_function["auc"].append(auc)
        results_activation_function["logloss"].append(logloss)
        results_activation_function["rmse"].append(rmse)
        
    print("\t\t-- DROPOUT RATES --\t\t")
    for dnn_dropout in dnn_dropout_list:
        print("\nTesting {dnn_dropout}...".format(dnn_dropout = dnn_dropout))
        
        model = DeepFM(linear_feature_columns,dnn_feature_columns, dnn_dropout = dnn_dropout, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_dropout["auc"].append(auc)
        results_dropout["logloss"].append(logloss)
        results_dropout["rmse"].append(rmse)

        
    print("\t\t-- HIDDEN UNITS --\t\t")
    for dnn_hidden_units in dnn_hidden_units_list:
        print("\nTesting {dnn_hidden_units}...".format(dnn_hidden_units = dnn_hidden_units))
        
        model = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units = dnn_hidden_units, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_number_of_neurons["auc"].append(auc)
        results_number_of_neurons["logloss"].append(logloss)
        results_number_of_neurons["rmse"].append(rmse)
    
    if PLOT:
        create_plots("DFM", "criteo", results_activation_function, "Activation Function", "activation_func", dnn_activation_list)
        create_plots("DFM", "criteo", results_dropout, "Dropout Rate", "dropout", dnn_dropout_list)
        create_plots("DFM", "criteo", results_number_of_neurons, "Number of Neurons per layer", "nr_neurons", dnn_hidden_units_list)

def test_PNN_avazu(data,train,test):

    print("\nTesting PNN on avazu dataset...\n")
    
    results_activation_function = {"auc": [], "logloss": [], "rmse": []}
    results_dropout = {"auc": [], "logloss": [], "rmse": []}
    results_number_of_neurons = {"auc": [], "logloss": [], "rmse": []}

    auc = 0
    logloss = 0
    rmse = 0
    
    features_labels = train.columns
        
    sparse_features_labels = features_labels[1:23]
    target_label = features_labels[0]
    
    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4,) for feat in sparse_features_labels]
    
    feature_names = get_feature_names(dnn_feature_columns)
    
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    
    true_y = test[target_label].values
    
    print("\t\t-- ACTIVATION FUNCTIONS --\t\t")
    for dnn_activation in dnn_activation_list:
        print("\nTesting {dnn_activation}...".format(dnn_activation = dnn_activation))
        
        # model = PNN(dnn_feature_columns, use_inner=False, use_outter=True, dnn_activation = dnn_activation, task='binary')
        model = PNN(dnn_feature_columns, use_inner=True, use_outter=False, dnn_activation = dnn_activation, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_activation_function["auc"].append(auc)
        results_activation_function["logloss"].append(logloss)
        results_activation_function["rmse"].append(rmse)
        
    print("\t\t-- DROPOUT RATES --\t\t")
    for dnn_dropout in dnn_dropout_list:
        print("\nTesting {dnn_dropout}...".format(dnn_dropout = dnn_dropout))
        
        # model = PNN(dnn_feature_columns, use_inner=False, use_outter=True, dnn_dropout = dnn_dropout, task='binary')
        model = PNN(dnn_feature_columns, use_inner=True, use_outter=False, dnn_dropout = dnn_dropout, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_dropout["auc"].append(auc)
        results_dropout["logloss"].append(logloss)
        results_dropout["rmse"].append(rmse)
        
    print("\t\t-- HIDDEN UNITS --\t\t")
    for dnn_hidden_units in dnn_hidden_units_list:
        print("\nTesting {dnn_hidden_units}...".format(dnn_hidden_units = dnn_hidden_units))
        
        # model = PNN(dnn_feature_columns, use_inner=False, use_outter=True, dnn_hidden_units = dnn_hidden_units, task='binary')
        model = PNN(dnn_feature_columns, use_inner=True, use_outter=False, dnn_hidden_units = dnn_hidden_units, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0,validation_split=TEST_PROPORTION )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_number_of_neurons["auc"].append(auc)
        results_number_of_neurons["logloss"].append(logloss)
        results_number_of_neurons["rmse"].append(rmse)
        
    if PLOT:
        # create_plots("OPNN", "avazu", results_activation_function, "Activation Function", "activation_func", dnn_activation_list)
        # create_plots("OPNN", "avazu", results_dropout, "Dropout Rate", "dropout", dnn_dropout_list)
        # create_plots("OPNN", "avazu", results_number_of_neurons, "Number of Neurons per layer", "nr_neurons", dnn_hidden_units_list)
        create_plots("PNN", "avazu", results_activation_function, "Activation Function", "activation_func", dnn_activation_list)
        create_plots("PNN", "avazu", results_dropout, "Dropout Rate", "dropout", dnn_dropout_list)
        create_plots("PNN", "avazu", results_number_of_neurons, "Number of Neurons per layer", "nr_neurons", dnn_hidden_units_list)

def test_FNN_avazu(data,train,test):
    
    print("\nTesting FNN on avazu dataset...\n")
    
    results_activation_function = {"auc": [], "logloss": [], "rmse": []}
    results_dropout = {"auc": [], "logloss": [], "rmse": []}
    results_number_of_neurons = {"auc": [], "logloss": [], "rmse": []}

    auc = 0
    logloss = 0
    rmse = 0    

    features_labels = train.columns
        
    sparse_features_labels = features_labels[1:23]
    target_label = features_labels[0]
    
    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4,) for feat in sparse_features_labels] 
    linear_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4,) for feat in sparse_features_labels]
 
    feature_names = get_feature_names(linear_feature_columns+ dnn_feature_columns)
    
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    
    true_y = test[target_label].values
    
    print("\t\t-- ACTIVATION FUNCTIONS --\t\t")
    for dnn_activation in dnn_activation_list:
        print("\nTesting {dnn_activation}...".format(dnn_activation = dnn_activation))
        
        model = FNN(linear_feature_columns, dnn_feature_columns, dnn_activation = dnn_activation, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_activation_function["auc"].append(auc)
        results_activation_function["logloss"].append(logloss)
        results_activation_function["rmse"].append(rmse)

        
    print("\t\t-- DROPOUT RATES --\t\t")
    for dnn_dropout in dnn_dropout_list:
        print("\nTesting {dnn_dropout}...".format(dnn_dropout = dnn_dropout))
        
        model = FNN(linear_feature_columns,dnn_feature_columns, dnn_dropout = dnn_dropout, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_dropout["auc"].append(auc)
        results_dropout["logloss"].append(logloss)
        results_dropout["rmse"].append(rmse)
        
    print("\t\t-- HIDDEN UNITS --\t\t")
    for dnn_hidden_units in dnn_hidden_units_list:
        print("\nTesting {dnn_hidden_units}...".format(dnn_hidden_units = dnn_hidden_units))
        
        model = FNN(linear_feature_columns, dnn_feature_columns, dnn_hidden_units = dnn_hidden_units, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_number_of_neurons["auc"].append(auc)
        results_number_of_neurons["logloss"].append(logloss)
        results_number_of_neurons["rmse"].append(rmse)
        
    if PLOT:
        create_plots("FNN", "avazu", results_activation_function, "Activation Function", "activation_func", dnn_activation_list)
        create_plots("FNN", "avazu", results_dropout, "Dropout Rate", "dropout", dnn_dropout_list)
        create_plots("FNN", "avazu", results_number_of_neurons, "Number of Neurons per layer", "nr_neurons", dnn_hidden_units_list)


def test_DFM_avazu(data,train,test):
    print("\nTesting DFM on avazu dataset...\n")
    
    results_activation_function = {"auc": [], "logloss": [], "rmse": []}
    results_dropout = {"auc": [], "logloss": [], "rmse": []}
    results_number_of_neurons = {"auc": [], "logloss": [], "rmse": []}

    auc = 0
    logloss = 0
    rmse = 0

    features_labels = train.columns
        
        
    sparse_features_labels = features_labels[1:23]
    target_label = features_labels[0]
    
    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4,) for feat in sparse_features_labels]
    linear_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4,) for feat in sparse_features_labels]
 
    feature_names = get_feature_names(linear_feature_columns+ dnn_feature_columns)
    
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    
    true_y = test[target_label].values
    
    print("\t\t-- ACTIVATION FUNCTIONS --\t\t")
    for dnn_activation in dnn_activation_list:
        print("\nTesting {dnn_activation}...".format(dnn_activation = dnn_activation))
        
        model = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_activation = dnn_activation, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_activation_function["auc"].append(auc)
        results_activation_function["logloss"].append(logloss)
        results_activation_function["rmse"].append(rmse)
        
    print("\t\t-- DROPOUT RATES --\t\t")
    for dnn_dropout in dnn_dropout_list:
        print("\nTesting {dnn_dropout}...".format(dnn_dropout = dnn_dropout))
        
        model = DeepFM(linear_feature_columns,dnn_feature_columns, dnn_dropout = dnn_dropout, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_dropout["auc"].append(auc)
        results_dropout["logloss"].append(logloss)
        results_dropout["rmse"].append(rmse)

        
    print("\t\t-- HIDDEN UNITS --\t\t")
    for dnn_hidden_units in dnn_hidden_units_list:
        print("\nTesting {dnn_hidden_units}...".format(dnn_hidden_units = dnn_hidden_units))
        
        model = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units = dnn_hidden_units, task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        model.fit(train_model_input, train[target_label].values, batch_size=256, epochs=10, verbose=0, validation_split=TEST_PROPORTION, )
        pred_y = model.predict(test_model_input, batch_size=256)
        
        auc = compute_auc(true_y, pred_y)
        logloss = compute_log_loss(true_y, pred_y)
        rmse = compute_rmse(true_y, pred_y)
        
        results_number_of_neurons["auc"].append(auc)
        results_number_of_neurons["logloss"].append(logloss)
        results_number_of_neurons["rmse"].append(rmse)
    
    if PLOT:
        create_plots("DFM", "avazu", results_activation_function, "Activation Function", "activation_func", dnn_activation_list)
        create_plots("DFM", "avazu", results_dropout, "Dropout Rate", "dropout", dnn_dropout_list)
        create_plots("DFM", "avazu", results_number_of_neurons, "Number of Neurons per layer", "nr_neurons", dnn_hidden_units_list)

def create_plots(algorithm, dataset_name, results, title, filename, x): 

    fig, ax = plt.subplots()
    ax.plot(x, results["auc"], marker = "s", color = "r")
    ax.plot(x, results["logloss"], marker = "s", color = "g")
    ax.plot(x, results["rmse"], marker = "s", color = "b")
    fig.legend(["AUC", "LogLoss", "RMSE"])
    fig.suptitle(title)
    ax.figure.savefig("./plots/{dataset_name}/{algorithm}/{filename}_{algorithm}.jpg".format(algorithm = algorithm, dataset_name = dataset_name, filename = filename))
    
    
if __name__ == "__main__":
    
    start_time = time.time()

    data_criteo, train_criteo, test_criteo = process_criteo()
    data_avazu, train_avazu, test_avazu = process_avazu()

    test_FNN_criteo(data_criteo,train_criteo,test_criteo)
    test_FNN_avazu(data_avazu,train_avazu,test_avazu)
    
    # test_PNN_criteo(data_criteo, train_criteo, test_criteo)
    # test_PNN_avazu(data_avazu,train_avazu,test_avazu)
    
    # test_DFM_criteo(data_criteo, train_criteo, test_criteo)
    # test_DFM_avazu(data_avazu,train_avazu,test_avazu)
        
    print("Program ended in {time} seconds.".format(time = round(time.time() - start_time, 2)))