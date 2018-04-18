# -*- coding: utf-8 -*-

import face_detector
import normalization
import gray_feature
import landmark_feature
import csv_helper
import random_forest
import recognition

import os
import numpy as np

if __name__ == "__main__":
    #第1步：检测人脸
    faces_folder_path = "/Users/Vivien/Documents/emotion_recognition/Emotion_Recognition/preprocess/origin"
    detector,predictor = face_detector.get_face_detector()

    #第2步：关键点对齐(仿射变换)
    normalization.normalize(faces_folder_path, detector, predictor)

    #第3步：提取块特征（灰度值）
    #ImageFilePath='/Users/Vivien/Documents/论文/论文实验/归一化/归一化结果数据集/Dataset/croped'
    ImageFilePath='/Users/Vivien/Documents/emotion_recognition/Emotion_Recognition/preprocess/cropped'
    block=8
    train_feature_path='/Users/Vivien/Documents/emotion_recognition/Emotion_Recognition/data_set/gray_feature.csv'
    gray_feature.save_train_gray_feature(ImageFilePath, block, train_feature_path)

    #第4步：提取点特征
    images_path = os.getcwd() + "/preprocess/detected"
    train_feature_path = os.getcwd() + "/data_set/landmark_feature.csv"
    landmark_feature.get_landmark_feature(images_path,train_feature_path)

    #第5步：整理训练数据集（标签）
    #仅调用一次 生成所有数据集合。
    entropy_file = os.getcwd() + '/data_set/gray_feature.csv'
    landmark_file = os.getcwd() + '/data_set/landmark_feature.csv'
    degree_file = os.getcwd() + "/data_set/degree.csv"
    all_data_file = os.getcwd() + "/data_set/all_data.csv"
    csv_helper.combine_all_feature(entropy_file, landmark_file, degree_file, all_data_file)
    #csv_helper.combine_landmark_feature(landmark_file, degree_file, all_data_file)


    #按比例随机生成
    train_file = os.getcwd() + "/data_set/train.csv"
    test_file = os.getcwd() + "/data_set/test.csv"
    csv_helper.spilt_test_data(all_data_file, train_file, test_file, 0.2)


    #第6步：训练整体识别模型(对比实验)
    dataset = csv_helper.load_csv(os.getcwd() + '/data_set/train.csv')
    classifyTag, dataset = csv_helper.clear_data(dataset, -8, -2)
    random_forest.generate_whole_classify_model(dataset)


    #第7步：训练单一识别模型
    #random_forest.generate_single_classify_model(classifyTag, dataset)


    #第8步：训练单一回归模型
    #random_forest.generate_single_degree_model(classifyTag, dataset)

    #第9步：测试集进入训练模型
    test_set = recognition.get_test_set()
    #classifier_models = recognition.get_classifier_model()
    #degree_models = recognition.get_degree_model()
    #classifier_res = recognition.get_multi_results(classifier_models, test_set)
    #degree_res = recognition.get_degree_result(degree_models, test_set)

    #第10步：阈值判断，输出结果
    #recognition.judge(classifier_res, degree_res)
    recognition.get_whole_classify(test_set)

    test_set = np.array(test_set)
    test_label = test_set[:, -1]
    print test_label
