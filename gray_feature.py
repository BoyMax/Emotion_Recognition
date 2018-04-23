# -*- coding: utf-8 -*-

import cv2
import numpy as np
from numpy import linalg as la
import os
import math
import csv_helper



#第3步：提取块特征

'''获取图片文件夹下 所有图片文件'''
def file_name(file_dir):
    imageFiles=[]
    imageNames=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                imageFiles.append(os.path.join(root, file))
                #imageName = file[0:6].replace(".", "-");
                imageName = file[0:6]
                imageNames.append(imageName)
    return imageFiles, imageNames


'''获取单张图片切分的感兴趣区域数组 行方向和列方向均分成n块'''
def ROI_block(imgPath,n):
    img =cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
    rois=[]
    height = img.shape[0]
    width = img.shape[1]
    heightStep = height/n
    widthStep = width/n
    index = 0
    for row in range(0,(height-1),heightStep):
        for col in range(0,(width-1),widthStep):
            roi = img[col:col+widthStep-1,row:row+heightStep-1]
            cv2.imshow('roi',roi)
            '''cv2.waitKey(0)'''
            rois.append(roi)
            index +=1
    return rois

'''计算每个矩阵的奇异熵'''
def cal_singular_entropy(matrix):
    U,S,VT = la.svd(matrix) #直接进行分解
    singular_value_array = S  #奇异值'''
    entropy=-1  #计算奇异值的熵
    if(len(singular_value_array)>0):
        result=0
    for singular_value in singular_value_array:
        if singular_value ==0:
            break
        singular_value_rate = singular_value/sum(singular_value_array)
        entropy += -singular_value_rate * math.log(singular_value_rate,2)
    return entropy


'''主函数：通过图像子块的奇异熵表示图像特征(参数：文件夹路径，分块数)'''
def feature_extraction(ImagePath,block):
    imageFiles,imageNames = file_name(ImagePath)
    imagePath = []
    imageIndex = 0
    features = np.zeros((len(imageFiles),(block+1)*(block+1)))
    for image in imageFiles:
        rois = ROI_block(image, block)
        imagePath.append(image)
        image_entropys = np.zeros((block+1)*(block+1))
        roiIndex = 0
        for roi in rois:
            entropy = cal_singular_entropy(roi)
            image_entropys[roiIndex] = entropy
            roiIndex +=1
        features[imageIndex]=image_entropys
        imageIndex +=1

    return imagePath,imageNames,features




def save_train_gray_feature(ImageFilePath,block,train_feature_path):
    imagePath,imageNames,features = feature_extraction(ImageFilePath,block)
    csv_helper.file_clear(train_feature_path)
    for i in range(len(imagePath)):
        feature_str =''
        for element in features[i].flat:
            if element != 0:
                feature_str = feature_str + str(element) +","
        #emotion = imageNames[i].split('.')[1][0:2]
        #text=imagePath[i] + " " +feature_str
        traindataline = feature_str+imageNames[i].replace(".", "-")
        csv_helper.insertLineIntoFile(train_feature_path,i+1,traindataline)


if __name__ == "__main__" :
    '''
    @param: 提供的参数：1，图像的文件名位置，2，需要保留的文件位置，3，图像的格式列表
    '''
    ImageFilePath='/Users/Vivien/Documents/emotion_recognition/Emotion_Recognition/preprocess/cropped'
    block=8
    train_feature_path='/Users/Vivien/Documents/emotion_recognition/Emotion_Recognition/data_set/gray_feature.csv'
    save_train_gray_feature(ImageFilePath,block,train_feature_path)
