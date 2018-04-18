# -*- coding: utf-8 -*-

import os
import dlib
import glob
import cv2
import csv_helper
import face_detector
import numpy as np
import math

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
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            shape = predictor(img, d)
            feature_str =''
            for i in range(68-1):
                if shape.part(i).x - shape.part(i+1).x == 0:
                    gradient = 0
                else:
                    gradient = float((shape.part(i).y - shape.part(i+1).y))/float((shape.part(i).x - shape.part(i+1).x))
                feature_str = feature_str + "%.2f" % gradient+","
                #feature_str = feature_str + "%d" % shape.part(i).x +","+"%d" % shape.part(i).y +","
            str = f.split('.jpg')[0].rsplit('/')
            picName = str[len(str)-1][0:6].replace(".", "-")
            landmark_feature_pic = feature_str + picName
            csv_helper.insertLineIntoFile(train_feature_path,index,landmark_feature_pic)
            index += 1
'''
            xlist = []
            ylist = []
            for i in range(1,68): #Store X and Y coordinates in two lists
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))
            xmean = np.mean(xlist) #Find both coordinates of centre of gravity
            ymean = np.mean(ylist)

            xcentral = [(x-xmean) for x in xlist] #Calculate distance centre <-> other points in both axes
            ycentral = [(y-ymean) for y in ylist]

            landmarks_vectorised = []

            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorised.append(w)
                landmarks_vectorised.append(z)
                meannp = np.asarray((ymean,xmean))
                coornp = np.asarray((z,w))
                dist = np.linalg.norm(coornp-meannp)
                #landmarks_vectorised.append(dist)
                #landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))

            feature_str =''
            for i in landmarks_vectorised:
                feature_str = feature_str + "%d" % i +","

            str = f.split('.jpg')[0].rsplit('/')
            picName = str[len(str)-1][0:6].replace(".", "-")
            landmark_feature_pic = feature_str + picName
            csv_helper.insertLineIntoFile(train_feature_path,index,landmark_feature_pic)
            index += 1
'''

if __name__ == "__main__":
    images_path = os.getcwd() + "/preprocess/cropped"
    train_feature_path = os.getcwd() + "/data_set/landmark_feature.csv"
    get_landmark_feature(images_path,train_feature_path)

