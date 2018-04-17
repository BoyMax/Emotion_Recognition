# -*- coding: utf-8 -*-
import csv_helper
import os
import numpy as np

from sklearn.externals import joblib


#获取测试集

def get_test_set():
    testset = csv_helper.load_csv(os.getcwd() + '/data_set/test.csv')
    csv_helper.clear_test_data(testset, -1)
    return testset


#加载7个判别模块
def get_classifier_model():
    classifier_models = dict()
    classifier_models["HA"] = joblib.load(os.getcwd() + '/model/HA_classify_model.m')
    classifier_models["SA"] = joblib.load(os.getcwd() + '/model/SA_classify_model.m')
    classifier_models["SU"] = joblib.load(os.getcwd() + '/model/SU_classify_model.m')
    classifier_models["DI"] = joblib.load(os.getcwd() + '/model/DI_classify_model.m')
    classifier_models["AN"] = joblib.load(os.getcwd() + '/model/AN_classify_model.m')
    classifier_models["FE"] = joblib.load(os.getcwd() + '/model/FE_classify_model.m')
    classifier_models["NE"] = joblib.load(os.getcwd() + '/model/NE_classify_model.m')
    return classifier_models


#得出多分类结果
def get_multi_results(classifier_models,test_set):
    test_set = np.array(test_set)
    enmotions = dict()

    for tag, model in classifier_models.items():
        enmotions[tag] = model.predict(test_set[:, 0:-8])

    return enmotions

#加载6个分量模块
def get_degree_model():
    degree_models = dict()
    degree_models["HA"] = joblib.load(os.getcwd() + '/model/HA_degree_model.m')
    degree_models["SA"] = joblib.load(os.getcwd() + '/model/SA_degree_model.m')
    degree_models["SU"] = joblib.load(os.getcwd() + '/model/SU_degree_model.m')
    degree_models["DI"] = joblib.load(os.getcwd() + '/model/DI_degree_model.m')
    degree_models["AN"] = joblib.load(os.getcwd() + '/model/AN_degree_model.m')
    degree_models["FE"] = joblib.load(os.getcwd() + '/model/FE_degree_model.m')
    return degree_models


#得出多分量结果
def get_degree_result(degree_models,test_set):
    test_set = np.array(test_set)
    degrees = dict()
    for tag, model in degree_models.items():
        degrees[tag] = model.predict(test_set[:, 0:-8])
    return degrees


#判定分量阈值
def judge(classifier_res, degree_res):
    pic_nums = len(classifier_res.values()[0])
    pic_emotions = list()
    #将数据拆分成单张图片处理：
    for pic_num in range(pic_nums):
        emotion = dict()
        degree = dict()
        class_value = 0
        for tag, class_res in classifier_res.items():
            emotion[tag] = class_res[pic_num]
            class_value += class_res[pic_num]
            if class_res[pic_num] == 1:
                class_item = tag
        if class_value == 1:
            print ('Pic Number {} is single emotion: {}'.format(pic_num,class_item))
            continue
        '''
        elif class_value == 0:
            print ('Pic Number {} is single emotion: NE'.format(pic_num))
            #continue
        '''

        #获取到每个基础表情分量：
        for tag, degree_vals in degree_res.items():
            degree[degree_vals[pic_num]] = tag
        #降序排列分量值：
        degree.keys().sort()
        sorted_degree = sorted(degree.keys(), reverse = True)
        #获取前两个值；
        first = sorted_degree[0]
        second = sorted_degree[1]
        #阈值判定：
        #--绝对值判定：
        if second > 4 :
            print ('Pic Number {} is complex emotion {} and {}, first:{}, second:{}'.format(pic_num,degree[first], degree[second], first, second))
        elif first < 3.3:
            print('Pic Number {} is single emotion: NORMAL'.format(pic_num))
        #--相对值判定：
        elif first>3.5 and first - second <0.5:
            print ('Pic Number {} is complex emotion {} and {}, first:{}, second:{}'.format(pic_num, degree[first], degree[second], first, second))
        else:
            print('Pic Number {} is single emotion {}, first:{}'.format(pic_num, degree[first],first))


if __name__ == "__main__":
    test_set = get_test_set()
    classifier_models = get_classifier_model()
    degree_models = get_degree_model()
    classifier_res = get_multi_results(classifier_models, test_set)
    degree_res = get_degree_result(degree_models, test_set)
    judge(classifier_res, degree_res)

    test_set = np.array(test_set)
    test_label = test_set[:, -1]
    print test_label

