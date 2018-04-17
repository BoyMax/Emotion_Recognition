# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


import numpy as np
import csv_helper


def get_singel_kind_set(dataset, tagNumber, tag, featureColumn, targetColumn):
    # 1. 准备训练数据（积极数据、消print 'Enter {tag} forest training.'.format(tag=tag)
    trainDataList = list()
    targetDataList = list()
    tagColumnIndex = len(dataset[0]) + targetColumn
    tagCounter = 0
    for row in dataset:
        if row[tagColumnIndex] == tagNumber:
            tagCounter += 1
            trainDataList.append(row[0:featureColumn])
            targetDataList.append(1)
    print '----- The number of positive data is {tagCounter}'.format(tagCounter=tagCounter)
    for row in dataset:
        if row[tagColumnIndex] != tagNumber and tagCounter > 0:
            tagCounter -= 1
            trainDataList.append(row[0:featureColumn])
            targetDataList.append(0)
    return trainDataList, targetDataList


def classifier_error_with_param(train_data, train_label):
    RANDOM_STATE = 10
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
         RandomForestClassifier(warm_start=True, oob_score=True,
                                max_features="sqrt",
                                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
         RandomForestClassifier(warm_start=True, max_features='log2',
                                oob_score=True,
                                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
         RandomForestClassifier(warm_start=True, max_features=None,
                                oob_score=True,
                                random_state=RANDOM_STATE))
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 1
    max_estimators = 200

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(train_data, train_label)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()



def regressor_error_with_param(train_data, train_degree,tag):
    RANDOM_STATE = 10
    ensemble_clfs = [
        ("RandomForestRegressor, max_features='sqrt'",
         RandomForestRegressor(warm_start=True, oob_score=True,
                                max_features="sqrt",
                                random_state=RANDOM_STATE)),
        ("RandomForestRegressor, max_features='log2'",
         RandomForestRegressor(warm_start=True, max_features='log2',
                                oob_score=True,
                                random_state=RANDOM_STATE)),
        ("RandomForestRegressor, max_features=None",
         RandomForestRegressor(warm_start=True, max_features=None,
                                oob_score=True,
                                random_state=RANDOM_STATE))
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 1
    max_estimators = 200

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(train_data, train_degree)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators of "+tag)
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    data_set = csv_helper.load_csv('/Users/Vivien/Desktop/dataset/TrainData/all_feature.csv')
    classifyTag = csv_helper.clear_data(data_set, -8, -2, True)

    trainDataList, targetDataList = get_singel_kind_set(data_set, 6, "nn", -8, -2)
    classifier_error_with_param(trainDataList, targetDataList)





'''
# randomforest方法
rf = RandomForestClassifier(oob_score=True, random_state=10, n_estimators=20, bootstrap=True, criterion='gini',
                            max_features='auto', min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_jobs=1, warm_start=False)
rf = rf.fit(train_data, train_label)
pre_rf = rf.predict(test_data)
print pre_rf
acc = float((pre_rf == test_label).sum()) / len(test_data)
print u'准确率：%f' % (acc)



  column_dic = {
        'HA': -8,
        'SA': -7,
        'SU': -6,
        'AN': -5,
        'DI': -4,
        'FE': -3
    }

    data_regression = np.array(data_set)
    for (tag, tagNumber) in classifyTag.items():
        train_data = data_regression[:, 0:-8]
        train_degree = data_regression[:, column_dic[tag]]
        regressor_error_with_param(train_data, train_degree,tag)

'''
