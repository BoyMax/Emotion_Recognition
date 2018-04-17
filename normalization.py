# -*- coding: utf-8 -*-

import os
import dlib
import glob
import cv2
import face_detector


#第1步：检测人脸
def face_detection(detector,file):
    img = cv2.imread(file)
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    print("Detected faces:",dets)
    for k,d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        leftTop = (int(d.left()), int(d.top()))
        rightBottom = (int(d.right()), int(d.bottom()))
        #cvRectangle函数参数： 图片， 左上角， 右下角， 颜色， 线条粗细， 线条类型，点类型
        #cv2.rectangle(img, leftTop, rightBottom, (0, 0, 255), 1, 4, 0)
        #cv2.imshow('face',img)
        #cv2.waitKey(0)


        #放大脸部区域
        a1=x1=int(d.left())
        a2=x2=int(d.right())
        b1=y1=int(d.top())
        b2=y2=int(d.bottom())

        rows = img.shape[0]
        cols = img.shape[1]

        a1= (x1-(x2-x1)/4) if (x1-(x2-x1)/4)>=0 else 0
        b1= (y1-(y2-y1)/4) if (y1-(y2-y1)/4)>=0 else 0
        a2= (x2+(x2-x1)/4) if (x2+(x2-x1)/4)<cols else cols
        b2= (y2+(y2-y1)/4) if (y2+(y2-y1)/4)<rows else rows

        #优化--裁剪出感兴趣区域（放大脸部区域后裁剪）：
        detected_region=img[b1:b2,a1:a2,:]
        detected_region=cv2.resize(detected_region, (256, 256))

        #cv2.imshow('detected_region', detected_region)
        #cv2.waitKey(0)

        #裁剪出感兴趣区域（直接裁剪）：
        #detected_region = img[d.top():d.bottom(), d.left():d.right()]
        #detected_region = cv2.resize(detected_region, (256, 256))

        str = file.split('/')
        saveImageName = str[len(str)-1]
        #print(saveImageName)
        cv2.imwrite(os.getcwd() + "/preprocess/detected/"+saveImageName, detected_region)
    return saveImageName, detected_region

#第2步：关键点对齐(仿射变换)
def face_landmarks(predictor, saveImageName, img, detector):
    new_dets = detector(img, 1)
    for k, d in enumerate(new_dets):
        shape = predictor(img, d)
        image = dlib.get_face_chip(img, shape)
        cv_bgr_img = cv2.resize(image, (256, 256))

        #输出旋转图片
        #cv2.imshow('aligned_pic',cv_bgr_img)
        #cv2.waitKey(0)
        cv2.imwrite(os.getcwd() +"/preprocess/aligned/"+saveImageName, cv_bgr_img)


        #缩小裁剪区域
        rows = cv_bgr_img.shape[0]
        cols = cv_bgr_img.shape[1]
        croped_img = cv_bgr_img[rows/8:rows*7/8,cols/8:cols*7/8,:]
        croped_img = cv2.resize(croped_img, (256, 256))
        #cv2.imshow('cropped_pic', croped_img)
        #cv2.waitKey(0)
        print(cv2.imwrite(os.getcwd() +"/preprocess/cropped/"+saveImageName, croped_img))


def normalize(faces_folder_path,detector,predictor):
    files = glob.glob(os.path.join(faces_folder_path, "*.jpg"))
    for file in files:
    #file = "/Users/Vivien/Documents/论文/论文实验/归一化/归一化结果数据集/Dataset/origin/KA.SA3.35.jpg"
        saveImageName, img = face_detection(detector,file)
        face_landmarks(predictor, saveImageName, img, detector)



if __name__ =="__main__":
    faces_folder_path = "/Users/Vivien/Documents/论文/论文实验/归一化/归一化结果数据集/Dataset/origin"
    detector,predictor = face_detector.get_face_detector()
    normalize(faces_folder_path, detector, predictor)


