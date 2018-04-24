# -*- coding: utf-8 -*-

import csv_helper
import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import svm

def train_random_forest(train_data, train_label):
    rf = RandomForestClassifier(oob_score=True,n_estimators=10)
    rf = rf.fit(train_data, train_label)
    os.chdir(os.getcwd() + '/model')
    model_name = "rf_model.m"
    joblib.dump(model, model_name)
    os.chdir(os.getcwd() + '/..')


def train_svm(train_data,  train_label):
    svc = svm.SVC(kernel='rbf', C=10)
    svc.fit(train_data, train_label)
    os.chdir(os.getcwd() + '/model')
    model_name = "svm_model.m"
    joblib.dump(model, model_name)
    os.chdir(os.getcwd() + '/..')



if __name__ == "__main__":
    train_set = csv_helper.load_csv(os.getcwd() + '/train/train_data.csv')
    csv_helper.clear_test_data(train_set, -1)
    data_set = np.array(train_set)

    train_data = data_set[:, 0: -1]
    train_label = data_set[:, -1]

    np.seterr(divide='ignore', invalid='ignore')
    #model = train_random_forest(train_data, train_label)
    model = train_svm(train_data, train_label)


