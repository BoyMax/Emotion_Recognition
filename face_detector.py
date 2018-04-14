# -*- coding: utf-8 -*-

import dlib

def get_face_detector():
    #人脸检测器：
    detector = dlib.get_frontal_face_detector()

    #关键点训练器：
    predictor_path="./model/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor
