# -*- coding: UTF-8 -*-

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import numpy as np

import os
import csv_helper


#targetColumn表示标签在数据行的倒数第targetColumn列 targetColumn= -2
#featureColumn表示标签在数据行的倒数第featureColumn列 featureColumn= -8
def make_classifier_model(dataset, tagNumber, tag, featureColumn, targetColumn):
    # 1. 准备训练数据（积极数据、消极数据对半）
    print 'Enter {tag} classify forest training.'.format(tag=tag)
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
    rf = RandomForestClassifier(oob_score=True, random_state=10, n_estimators=60)
    rf = rf.fit(trainDataList, targetDataList)
    print '----- The oob score is: {oob_score}'.format(oob_score=rf.oob_score_)
    return rf


#训练表情分量
#targetColumn表示表情种类标签在数据行的倒数第targetColumn列 targetColumn= -2
#degreeColumn表示分量在数据行的倒数第degreeColumn列 如快乐列degreeColumn= -8
#featureColumn表示标签在数据行的倒数第featureColumn列 featureColumn= -8
def make_degree_model(dataset,tagNumber,tag,featureColumn,targetColumn,degreeColumn):
    # 1. 准备训练数据（积极数据、消极数据对半）
    print 'Enter {tag} degree forest training.'.format(tag=tag)
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
    rf = RandomForestRegressor(oob_score=True, random_state=200, n_estimators=100)
    rf = rf.fit(trainDataList, targetDataList)
    print '----- The oob score is: {oob_score}'.format(oob_score=rf.oob_score_)
    return rf

def generate_single_classify_model(classifyTag, dataset):
    try:
        os.chdir(os.getcwd() + '/model')
        for (tag, tagNumber) in classifyTag.items():
            rf = make_classifier_model(dataset, tagNumber, tag, -8, -2)
            model_name = tag+"_classify_model.m"
            joblib.dump(rf, model_name)
            #print rf.predict(testset)
        #处理完成后，返回上级目录
        os.chdir(os.getcwd() + '/..')
    except Exception,err:
        print 'Error: ', err


def generate_whole_classify_model(dataset):

    data_set = np.array(dataset)

    train_data = data_set[:, 0: -8]
    train_label = data_set[:, -2]
    rf = RandomForestClassifier(oob_score=True,n_estimators=100)
    rf = rf.fit(train_data, train_label)
    os.chdir(os.getcwd() + '/model')
    model_name = "classify_model.m"
    joblib.dump(rf, model_name)
    os.chdir(os.getcwd() + '/..')
'''
    feature_importance = rf.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    fi_threshold = 50
    important_idx = np.where(feature_importance > fi_threshold)[0]
    important_features = important_idx
    print "\n", important_features.shape[0], "Important features(>", \
         fi_threshold, "% of max importance)...\n"#, \
         #important_features
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
    #get the figure about important features
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.title('Feature Importance')
    plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], \
         color='r',align='center')
    plt.yticks(pos, important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.draw()
    plt.show()



横向特征重要性的图
    f, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(rf.feature_importances_)),rf.feature_importances_)
    ax.set_title("Feature Importances")
    f.show()
    plt.show()
'''




def generate_single_degree_model(classifyTag, dataset):
    try:
        print(os.getcwd())
        os.chdir(os.getcwd() + '/model')
        for (tag, tagNumber) in classifyTag.items():
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
            if tag !="NE":
                rf_degree = make_degree_model(dataset,tagNumber, tag, -8, -2,degreeColumn)
                model_name = tag+"_degree_model.m"
                joblib.dump(rf_degree, model_name)
               #print rf_degree.predict(testset)
        os.chdir(os.getcwd() + '/..')
    except Exception,err:
        print 'Error: ', err



if __name__ == "__main__":
    dataset = csv_helper.load_csv(os.getcwd() + '/data_set/train.csv')
    classifyTag, dataset = csv_helper.clear_data(dataset, -8, -2)
    #generate_single_classify_model(classifyTag, dataset)
    #generate_single_degree_model(classifyTag, dataset)
    generate_whole_classify_model(dataset)


    #整体随机森林
'''
    try:
        dataset = csv_helper.load_csv(os.getcwd() + '/data_set/train_data.csv')
        all_data= np.array(dataset)
        classifyTag = clear_data(dataset,-8,-2,True)

        testset = csv_helper.load_csv('/Users/Vivien/Desktop/dataSet/TrainData/test.csv')
        clear_data(testset,-8,-2,True)

        rf = RandomForestClassifier(oob_score=True, random_state=10, n_estimators=100)
        rf = rf.fit(all_data[:, 0: -8], all_data[:, -2])
        print '----- The oob score of multi-classification is: {oob_score}'.format(oob_score=rf.oob_score_)
    except Exception,err:
        print 'Error: ', err
'''
