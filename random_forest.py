# -*- coding: UTF-8 -*-

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from csv import reader
import numpy as np

import os
import csvHelper

#featureColumn 表示标签在数据的倒数第featureColumn行 featureColumn=-8
#targetColumn表示标签在数据行的倒数第targetColumn列 targetColumn= -2
def clear_data(dataset,featureColumn,targetColumn,isDiscrete):
    # 转换feature 8表示最后8列不属于特征
    for row in dataset:
        for column in range(0, len(row) + featureColumn):
           row[column] = float(row[column].strip())
    # 离散类型转换tag
    if isDiscrete:
        column = len(dataset[0]) + targetColumn
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
           lookup[value] = i
        for row in dataset:
           row[column] = lookup[row[column]]
        return lookup

#targetColumn表示标签在数据行的倒数第targetColumn列 targetColumn= -2
#featureColumn表示标签在数据行的倒数第featureColumn列 featureColumn= -8
def make_classifier_model(dataset, tagNumber, tag, featureColumn, targetColumn):
    # 1. 准备训练数据（积极数据、消极数据对半）
    print 'Enter {tag} forest training.'.format(tag=tag)
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
    # 2. 进行训练，查看指标
    rf = RandomForestClassifier(oob_score=True, random_state=10, n_estimators=30)
    rf = rf.fit(trainDataList, targetDataList)
    print '----- The oob score is: {oob_score}'.format(oob_score=rf.oob_score_)
    return rf


#训练表情分量
#targetColumn表示表情种类标签在数据行的倒数第targetColumn列 targetColumn= -2
#degreeColumn表示分量在数据行的倒数第degreeColumn列 如快乐列degreeColumn= -8
#featureColumn表示标签在数据行的倒数第featureColumn列 featureColumn= -8
def make_degree_model(dataset,tagNumber,tag,featureColumn,targetColumn,degreeColumn):
    # 1. 准备训练数据（积极数据、消极数据对半）
    print 'Enter {tag} forest training.'.format(tag=tag)
    trainDataList = list()
    targetDataList = list()
    tagColumnIndex = len(dataset[0]) + targetColumn
    tagCounter = 0
    for row in dataset:
        if row[tagColumnIndex] == tagNumber:
            tagCounter += 1
            trainDataList.append(row[0:featureColumn])
            targetDataList.append(row[len(dataset[0])+degreeColumn])
    print '----- The number of positive data is {tagCounter}'.format(tagCounter=tagCounter)
    for row in dataset:
        if row[tagColumnIndex] != tagNumber and tagCounter > 0:
            tagCounter -= 1
            trainDataList.append(row[0:featureColumn])
            targetDataList.append(row[len(dataset[0])+degreeColumn])
    # 2. 进行训练，查看指标
    rf = RandomForestRegressor(oob_score=True, random_state=200, n_estimators=50)
    rf = rf.fit(trainDataList, targetDataList)
    print '----- The oob score is: {oob_score}'.format(oob_score=rf.oob_score_)
    return rf





if __name__ == "__main__":
    try:
        dataset = csvHelper.load_csv(os.getcwd() + '/data_set/train_data.csv')
        all_data= np.array(dataset)
        classifyTag = clear_data(dataset,-8,-2,True)

        #testset = csvHelper.load_csv('/Users/Vivien/Desktop/dataSet/TrainData/test.csv')
        #clear_data(testset,-8,-2,True)

#        rf = RandomForestClassifier(oob_score=True, random_state=10, n_estimators=100)
#        rf = rf.fit(all_data[:, 0: -8], all_data[:, -2])
#        print '----- The oob score of multi-classification is: {oob_score}'.format(oob_score=rf.oob_score_)

        os.chdir("/Users/Vivien/Desktop/Facial expression recognition/model")
        for (tag, tagNumber) in classifyTag.items():
#           rf = make_classifier_model(dataset, tagNumber, tag,-8,-2)
#            model_name = tag+"_model.m"
#            joblib.dump(rf, model_name)
            #print rf.predict(testset)

            degreeColumn = 0
            if tag=="HA":
                degreeColumn = -8
            elif tag=="SA":
                degreeColumn = -7
            elif tag=="SU":
                degreeColumn = -6
            elif tag=="AN":
                degreeColumn = -5
            elif tag=="DI":
                degreeColumn = -4
            elif tag=="FE":
                degreeColumn = -3
            if tag !="NE" and tag != "FE":
                rf_degree = make_degree_model(dataset,tagNumber, tag,-8,-2,degreeColumn)
              #  model_name = tag+"_degree_model.m"
              #  joblib.dump(rf_degree, model_name)
              #  print rf_degree.predict(testset)

    except Exception,err:
        print 'Error: ', err
