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
        leftTop = (int(d.left()),int(d.top()))
        rightBottom = (int(d.right()), int(d.bottom()))
        #cvRectangle函数参数： 图片， 左上角， 右下角， 颜色， 线条粗细， 线条类型，点类型
        #cv2.rectangle(img, leftTop, rightBottom, (0, 0, 255), 1, 4, 0)
        #cv2.imshow('face',img)
        #cv2.waitKey(0)

        #裁剪出感兴趣区域：
        detected_region = img[d.top():d.bottom(), d.left():d.right()]
        detected_region = cv2.resize(detected_region, (256, 256))

        str = file.split('/')
        saveImageName = str[len(str)-1]
        #print(saveImageName)
        cv2.imwrite(os.getcwd() + "/preprocess/detected/"+saveImageName, detected_region)
    return saveImageName, img, dets

#第2步：关键点对齐(仿射变换)
def face_landmarks(predictor, saveImageName, img, dets, detector):
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        image = dlib.get_face_chip(img, shape)
        cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #输出旋转图片
        #cv2.imshow('landmarks_pic',cv_bgr_img)
        #cv2.waitKey(0)

        #裁剪中间区域，避免旋转造成的锯齿边缘：先检测旋转归一后的感兴趣区域，根据矩形框裁剪,new_dets是旋转后的检测人脸框
        new_dets = detector(cv_bgr_img, 1)

        #确定只有一张脸：
        new_d = new_dets[0]
        croped_img = cv_bgr_img[new_d.top():new_d.bottom(), new_d.left():new_d.right()]
        croped_img = cv2.resize(croped_img, (256, 256))
        cv2.imshow('cropped_pic', croped_img)
        cv2.waitKey(0)
        print(cv2.imwrite(os.getcwd() +"/preprocess/cropped/"+saveImageName, croped_img))


#第3步：提取块特征（灰度值）


#第4步：提取点特征


#第5步：整理训练数据集（标签）

#第6步：训练整体识别模型

#第7步：训练单一识别模型

#第8步：训练单一回归模型

#第9步：判定分量差的阈值

#第10步：输出结果




def normalize(faces_folder_path,detector,predictor):
    files = glob.glob(os.path.join(faces_folder_path, "*.jpg"))
    for file in files:
        saveImageName, img, dets = face_detection(detector,file)
        face_landmarks(predictor, saveImageName, img, dets, detector)



if __name__ =="__main__":
    faces_folder_path = "/Users/Vivien/Documents/论文/论文实验/归一化/归一化结果数据集/Dataset/origin"
    detector,predictor = face_detector.get_face_detector()
    normalize(faces_folder_path, detector, predictor)
