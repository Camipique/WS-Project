# -*- coding: utf-8 -*-
"""
@author: Carlos
"""

from datetime import datetime

import pandas as pd
import tensorflow as tf

from sklearn.metrics import log_loss, roc_auc_score
from deepctr.models import PNN, DeepFM
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# if __name__ == "__main__":
#     start_time = datetime.now()
    
#     # read criteo data from file
#     data = pd.read_csv('./criteo_sample.txt')
    
#     # lists with the labels of the sparse (categorical) and dense (numerical) features
#     sparse_features_labels = ['C' + str(i) for i in range(1, 27)]
#     dense_features_labels = ['I' + str(i) for i in range(1, 14)]
    
#     # Pre-Processing - fill empty fields
#     data[sparse_features_labels] = data[sparse_features_labels].fillna('-1', )
#     data[dense_features_labels] = data[dense_features_labels].fillna(0, )
#     target = ['label']
    
#     sparse_features_data = data[sparse_features_labels]
#     dense_features_data = data[dense_features_labels]
    
#     # One-Hot encoding categorical features
#     # ohe = pd.get_dummies(sparse_features_data)
    
#     # Process data so it can be used by DeepCTR models
#     dnn_feature_columns =  [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4) for feat in sparse_features_labels] + [DenseFeat(feat, 1, ) for feat in dense_features_labels]
#     feature_names = get_feature_names(dnn_feature_columns)
    
#     # data = pd.concat([dense_features_data, ohe], axis=1)
    
#     train, test = train_test_split(data, test_size=0.2)
    
#     train_model_input = {name:train[name] for name in feature_names}
    
    
#     model = PNN(dnn_feature_columns=dnn_feature_columns)
#     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['binary_crossentropy'], )
#     model.fit(train_model_input, train[target].values, epochs=10, verbose=2, validation_split=0.2,)
    
#     print("Program ended in {time}".format(time = datetime.now() - start_time))

if __name__ == "__main__":
    start_time = datetime.now()
    print("Reading data...\n")
    
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']
    
    # print("One-hot enconding categorical features...\n")
    # ohe = pd.get_dummies(data[sparse_features])
    
    # sparse_features = [col for col in ohe.columns]
    
    # data_ohe = pd.concat([data[target], data[dense_features], ohe[sparse_features]], axis=1)
    
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    
    print("Normalizing numerical features...\n")
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    # data_ohe[dense_features] = mms.fit_transform(data[dense_features])

    
    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4,)
                            for feat in sparse_features] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    # fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data_ohe[feat].nunique(),embedding_dim=4,)
    #                         for feat in sparse_features] + [DenseFeat(feat, 1,)
    #                       for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    # train, test = train_test_split(data, test_size=0.2)
    print("Spltting dataset into train and test sets...\n")
    train, test = train_test_split(data, test_size=0.2)
    # train, test = train_test_split(data_ohe, test_size=0.2)
    
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    print("Defining PNN model...\n")
    model = PNN(dnn_feature_columns, task='binary')
    print("Compiling PNN model...\n")
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )
    print("Training the model...\n")
    model.fit(train_model_input, train[target].values,
              batch_size=256, epochs=10, verbose=1, validation_split=0.2, )
        
    print("\nTesting the model...\n")
    pred = model.predict(test_model_input, batch_size=256)
    
    print("test LogLoss", round(log_loss(test[target].values, pred), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred), 4))
    
    print("\nProgram ended in {time}".format(time = datetime.now() - start_time))