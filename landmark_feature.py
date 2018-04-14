# -*- coding: utf-8 -*-

import os
import dlib
import glob
import cv2
import csv_helper
import face_detector

#第4步：提取点特征

def get_landmark_feature(images_path,train_feature_path):
    detector, predictor = face_detector.get_face_detector()
    csv_helper.file_clear(train_feature_path)
    for f in glob.glob(os.path.join(images_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = cv2.imread(f)

        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        index = 1
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            shape = predictor(img, d)
            feature_str =''
            for i in range(68):
                feature_str = feature_str + "%d" % shape.part(i).x +","+"%d" % shape.part(i).y +","
            str = f.split('.jpg')[0].rsplit('/')
            picName = str[len(str)-1][0:6].replace(".", "-")
            landmark_feature_pic = feature_str + picName
            csv_helper.insertLineIntoFile(train_feature_path,index,landmark_feature_pic)
            index += 1



if __name__ == "__main__":
    images_path = os.getcwd() + "/preprocess/cropped"
    train_feature_path = os.getcwd() + "/data_set/landmark_feature.csv"
    get_landmark_feature(images_path,train_feature_path)

